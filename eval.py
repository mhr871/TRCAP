import argparse
import json
import os

import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import torch
from torch.utils.data import DataLoader

from Datasets.coco import COCOKarpathyTest
from Datasets.flickr import FlickrTest
from Datasets.dataset_utils import getTestTransforms
from Datasets.tasviret import TasvirEtTest
from Model import TRCaptionNetpp

from utils import over_write_args


@torch.no_grad()
def predict(model, data_loader, device, return_diagnostics=False, num_examples=5):
    # evaluate
    model.eval()
    result = []
    caption_lengths = []
    eos_count = 0
    total_count = 0
    sample_captions = []

    counter = 0
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
            counter += 1
    if return_diagnostics:
        diagnostics = {
            "avg_caption_len": sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0.0,
            "eos_rate": eos_count / total_count if total_count else 0.0,
            "sample_captions": sample_captions,
        }
        return result, diagnostics
    return result


def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file)

    img_ids = cocoRes.getImgIds()
    gts = {}
    res = {}
    for img_id in img_ids:
        gts[img_id] = coco.imgToAnns[img_id]
        res[img_id] = cocoRes.imgToAnns[img_id]

    print('tokenization...')
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    print('setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    result = {}
    for scorer, method in scorers:
        print('computing %s score...' % scorer.method())
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                result[m] = float(sc)
                print("%s: %0.3f" % (m, sc))
        else:
            result[method] = float(score)
            print("%s: %0.3f" % (method, score))

    print('SPICE: skipped')
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


def test(opt):
    print(opt)

    # initialize model
    model = TRCaptionNetpp(opt.model)

    checkpoint = torch.load(opt.weights, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(opt.device)
    model.eval()

    test_transforms = getTestTransforms(model_config=opt.model)

    if opt.dataset.lower() == 'coco':
        test_dataset = COCOKarpathyTest(dataset_root=opt.test_data,
                                        json_path=opt.test_json,
                                        transforms=test_transforms)
    elif opt.dataset.lower() == 'tasviret':
        test_dataset = TasvirEtTest(dataset_root=opt.test_data,
                                    json_path=opt.test_json,
                                    transforms=test_transforms)
    elif opt.dataset.lower() == 'flickr':
        test_dataset = FlickrTest(dataset_root=opt.test_data,
                                  json_path=opt.test_json,
                                  transforms=test_transforms)
    else:
        raise Exception()

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
    # os.remove(result_file)
    print(result)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TR-CLIP-Captioning!')
    parser.add_argument('--config', type=str, default='./configs/tasviret/tasviretpp_large_tasviret.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--weights', type=str, default='experiments/tasviretpp_large_tasviret_baseline/model_best.pth')
    parser.add_argument('--test-json', type=str, default='Data/tasvir-et/tasvir_test.json')
    parser.add_argument('--test-data', type=str, default='Data/flickr8k/images')
    parser.add_argument('--dataset', type=str, default='tasviret')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-worker', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='eval_outputs/tasviret_test')
    parser.add_argument('--prediction-file', type=str, default='predictions.json')
    parser.add_argument('--result-file', type=str, default='metrics.json')
    args = parser.parse_args()
    over_write_args(args, args.config)
    test(args)
