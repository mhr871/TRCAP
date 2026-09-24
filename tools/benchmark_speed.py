"""Diagnose training-throughput drops in this framework (proj_exp).

Originally written when training throughput dropped from ~5.3 it/s (earlier
encoder/decoder/projection experiments) to ~1.37 it/s here, even after
cutting num_workers 8 -> 2 (which not helping already argued against a
dataloader/CPU bottleneck). That drop was confirmed to come from
trainer.py's eval() doing more work per checkpoint than before -- mainly
METEOR scoring, which shells out to a slow Java subprocess and has since
been disabled again in eval.py for exactly this reason. If tqdm's reported
it/s is an average since the run started (rather than the instantaneous
recent rate), a single slow eval cycle early in training can drag that
average down for a long time even though the actual per-step training speed
between eval checkpoints is unaffected.

This script isolates and times, on the real GPU/config in use, exactly
where the time goes: pure training iterations (no eval at all), then one
full eval cycle broken into generation / BLEU+ROUGE+CIDEr scoring /
compute_val_loss's extra pass -- useful again if throughput regresses for
any other reason in the future.

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
from utils import over_write_args


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
    print(f"BLEU + ROUGE + CIDEr scoring (combined, METEOR disabled): {t3 - t2:.1f}s")

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
