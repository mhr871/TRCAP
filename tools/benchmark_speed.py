"""Diagnose a training-throughput drop between this framework (proj_exp)
and the earlier encoder/decoder/projection experiments' Colab runs.

The model/encoder/decoder are nearly identical between the two, so a large
it/s gap (reported: ~5.3 it/s before vs ~1.37 it/s here, even after cutting
num_workers 8 -> 2) is very unlikely to be a per-iteration forward/backward
regression -- reducing num_workers not helping already argues against a
dataloader/CPU bottleneck. The most likely cause is that trainer.py's eval()
now does noticeably more work per checkpoint than before:
  - METEOR scoring was re-added (eval.py); it shells out to a Java
    subprocess and is well known to be slow -- the original codebase
    excluded it for exactly this reason (see eval.py's comment on why
    METEOR was originally skipped).
  - compute_val_loss() is new: a full extra forward pass over the entire
    validation split.
If tqdm's reported it/s is an average since the run started (rather than
the instantaneous recent rate), a single slow eval cycle early in training
can drag that average down for a long time even though the actual per-step
training speed between eval checkpoints is unaffected.

This script isolates and times, on the real GPU/config in use, exactly
where the time goes: pure training iterations (no eval at all), then one
full eval cycle broken into generation / BLEU+ROUGE+CIDEr / METEOR alone /
compute_val_loss's extra pass.

Usage:
  python tools/benchmark_speed.py --config configs/projection_exp/P1_linear.yaml
  python tools/benchmark_speed.py --config configs/projection_exp/P1_linear.yaml --train-iters 100
"""
import argparse
import json
import os
import time
from argparse import Namespace

import torch
from torch.utils.data import DataLoader, RandomSampler

from Datasets.dataset_utils import getTestTransforms, getTrainDataset, getTestDataset
from Datasets.tasviret import TasvirEtTrain
from Model import TRCaptionNetPP
from eval import evaluate_on_coco_caption, predict
from utils import SafePTBTokenizer, over_write_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--train-iters', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    cli = parser.parse_args()

    args = Namespace(config=cli.config)
    over_write_args(args, cli.config)
    device = torch.device(cli.device)

    print(f"config: {cli.config}")
    print(f"model: {args.model}")
    print(f"batch_size={args.batch_size} num_workers={args.num_workers}\n")

    model = TRCaptionNetPP(args.model).to(device)
    model.train()
    if not args.model.get('train_decoder', True):
        model.language_decoder.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    train_dataset = getTrainDataset(args.train_dataset_root, args.train_json_path, model_config=args.model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=RandomSampler(train_dataset, replacement=True,
                                                    num_samples=(cli.train_iters + 5) * args.batch_size),
                              pin_memory=True, drop_last=True)

    print(f"=== 1) Pure training throughput ({cli.train_iters} iterations, no eval at all) ===")
    it = iter(train_loader)
    # Warmup batch (excluded from timing): pays for cuDNN autotune / lazy CUDA init / first-batch page faults.
    image, caption, _ = next(it)
    image = image.to(device)
    loss, _ = model(image, caption, return_acc=True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(cli.train_iters):
        image, caption, _ = next(it)
        image = image.to(device)
        loss, _ = model(image, caption, return_acc=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    pure_it_s = cli.train_iters / elapsed
    print(f"pure train: {cli.train_iters} it in {elapsed:.1f}s -> {pure_it_s:.2f} it/s\n")

    print("=== 2) Eval cycle breakdown (one full validation pass, same as trainer.py's eval()) ===")
    test_dataset = getTestDataset(args.test_dataset_root, args.val_json_path, model_config=args.model)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True, shuffle=False)

    model.eval()
    t0 = time.time()
    val_result, _ = predict(model, test_loader, device, return_diagnostics=True)
    t1 = time.time()
    print(f"generation (predict, beam search over {len(test_dataset)} images): {t1 - t0:.1f}s")

    os.makedirs('/tmp/benchmark_speed', exist_ok=True)
    pred_file = '/tmp/benchmark_speed/prediction.json'
    json.dump(val_result, open(pred_file, 'w'))

    t2 = time.time()
    evaluate_on_coco_caption(pred_file, args.val_json_path, logger_fn=lambda *a, **k: None)
    t3 = time.time()
    print(f"BLEU + METEOR + ROUGE + CIDEr scoring (combined): {t3 - t2:.1f}s")

    # Isolate METEOR alone, since it's the prime suspect.
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocotools.coco import COCO
    coco = COCO(args.val_json_path)
    cocoRes = coco.loadRes(pred_file)
    img_ids = cocoRes.getImgIds()
    gts = {i: coco.imgToAnns[i] for i in img_ids}
    res = {i: cocoRes.imgToAnns[i] for i in img_ids}
    tokenizer = SafePTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)
    t4 = time.time()
    Meteor().compute_score(gts_tok, res_tok)
    t5 = time.time()
    print(f"  of which METEOR alone (incl. JVM startup): {t5 - t4:.1f}s")

    val_loss_dataset = TasvirEtTrain(dataset_root=args.test_dataset_root, json_path=args.val_json_path,
                                     transforms=getTestTransforms(model_config=args.model))
    val_loss_loader = DataLoader(val_loss_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_memory=True, shuffle=False)
    t6 = time.time()
    with torch.no_grad():
        for image, caption, _ in val_loss_loader:
            image = image.to(device)
            model(image, caption, return_acc=False)
    t7 = time.time()
    print(f"compute_val_loss() extra pass ({len(val_loss_dataset)} samples): {t7 - t6:.1f}s")

    total_eval = t7 - t0
    lost_iters = total_eval * pure_it_s
    print(f"\nTOTAL eval cycle: {total_eval:.1f}s (~{total_eval / 60:.1f} min)")
    print(f"this runs once every num_eval_iter={args.num_eval_iter} training iterations")
    print(f"at the pure-train speed above, this eval cycle costs the equivalent of "
         f"~{lost_iters:.0f} training iterations")
    print(f"predicted average it/s over one full num_eval_iter window: "
         f"{args.num_eval_iter / (args.num_eval_iter / pure_it_s + total_eval):.2f}")


if __name__ == '__main__':
    main()
