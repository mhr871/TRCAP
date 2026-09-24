import argparse
import json
import os

import torch
from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from torch.utils.data import DataLoader

from Datasets.dataset_utils import getTestTransforms
from Datasets.tasviret import TasvirEtTest
from Model import TRCaptionNetPP
from utils import SafePTBTokenizer, over_write_args


@torch.no_grad()
def predict(model, data_loader, device, return_diagnostics=False, num_examples=5):
    # evaluate
    model.eval()
    result = []
    caption_lengths = []
    eos_count = 0
    total_count = 0
    sample_captions = []

    for image, img_ids in data_loader:
        image = image.to(device)
        if return_diagnostics:
            preds, token_ids = model.generate(image, return_token_ids=True)
            eos_token_id = model.tokenizer.sep_token_id
            eos_count += (token_ids == eos_token_id).any(dim=1).sum().item()
            total_count += token_ids.size(0)
            caption_lengths.extend(len(pred.split()) for pred in preds)
            sample_slots = max(0, num_examples - len(sample_captions))
            sample_captions.extend(preds[:sample_slots])
        else:
            preds = model.generate(image)
        for pred, img_id in zip(preds, img_ids):
            result.append({"image_id": int(img_id), "caption": pred})
    if return_diagnostics:
        diagnostics = {
            "avg_caption_len": sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0.0,
            "eos_rate": eos_count / total_count if total_count else 0.0,
            "sample_captions": sample_captions,
        }
        return result, diagnostics
    return result


def evaluate_on_coco_caption(res_file, label_file, outfile=None, logger_fn=print):
    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file)

    img_ids = cocoRes.getImgIds()
    gts = {}
    res = {}
    for img_id in img_ids:
        gts[img_id] = coco.imgToAnns[img_id]
        res[img_id] = cocoRes.imgToAnns[img_id]

    logger_fn('tokenization...')
    tokenizer = SafePTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    logger_fn('setting up scorers...')
    # METEOR, like the PTB tokenizer above, shells out to a bundled Java
    # subprocess (pycocoevalcap's own meteor-1.5.jar) rather than running in
    # pure Python, so a missing/misconfigured JVM can make it fail on its
    # own without affecting Bleu/Rouge/CIDEr. Per this project's "Son Kontrol
    # ve Calistirma Kurallari": if METEOR fails, log the error, record
    # `METEOR: null` (not 0.0 -- a missing measurement is not the same as a
    # measured zero), and keep training; every other scorer still falls back
    # to 0.0 on its own failure, same as before. SPICE is excluded entirely
    # (also Java, heavier still, and not in the experiment table this
    # project reports against).
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    result = {}
    for scorer, method in scorers:
        logger_fn('computing %s score...' % scorer.method())
        try:
            score, scores = scorer.compute_score(gts, res)
        except Exception as exc:
            is_meteor = scorer.method() == "METEOR"
            logger_fn(f"[ERROR] {scorer.method()} hesaplanamadi (Java/bagimlilik sorunu olabilir), "
                     f"egitime devam ediliyor: {exc}")
            for name in (method if type(method) == list else [method]):
                result.setdefault(name, None if is_meteor else 0.0)
            continue
        if type(method) == list:
            for sc, m in zip(score, method):
                result[m] = float(sc)
                logger_fn("%s: %0.3f" % (m, sc))
        else:
            result[method] = float(score)
            logger_fn("%s: %0.3f" % (method, score))

    logger_fn('SPICE: skipped (Java dependency, not required by the experiment table)')
    if not outfile:
        logger_fn(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


def test(opt):
    print(opt)

    # initialize model
    model = TRCaptionNetPP(opt.model)

    checkpoint = torch.load(opt.weights, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(opt.device)
    model.eval()

    test_transforms = getTestTransforms(model_config=opt.model)
    test_dataset = TasvirEtTest(dataset_root=opt.test_data, json_path=opt.test_json, transforms=test_transforms)

    test_loader = DataLoader(test_dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.num_workers,
                             pin_memory=True,
                             shuffle=False)

    test_result, diagnostics = predict(model, test_loader, opt.device, return_diagnostics=True)

    os.makedirs(opt.output_dir, exist_ok=True)
    result_file = os.path.join(opt.output_dir, opt.prediction_file)
    json.dump(test_result, open(result_file, 'w'))

    result = evaluate_on_coco_caption(result_file,
                                      opt.test_json,
                                      os.path.join(opt.output_dir, opt.result_file))
    result['avg_caption_len'] = diagnostics['avg_caption_len']
    result['eos_rate'] = diagnostics['eos_rate']
    with open(os.path.join(opt.output_dir, opt.result_file), 'w') as fp:
        json.dump(result, fp, indent=4)
    print(f"avg_caption_len: {diagnostics['avg_caption_len']:.3f}")
    print(f"eos_rate: {diagnostics['eos_rate']:.3f}")
    for index, caption in enumerate(diagnostics['sample_captions'], start=1):
        print(f"sample_caption_{index}: {caption}")
    print(result)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRCaptionNet++ Projection Adapter deneyleri -- eval')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--test-json', type=str, default='Data/tasvir-et/tasvir_test.json')
    parser.add_argument('--test-data', type=str, default='Data/flickr8k/images')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='eval_outputs/tasviret_test')
    parser.add_argument('--prediction-file', type=str, default='predictions.json')
    parser.add_argument('--result-file', type=str, default='metrics.json')
    args = parser.parse_args()
    over_write_args(args, args.config)
    test(args)
