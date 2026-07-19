import argparse
import json
import os

import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import torch
from torch.utils.data import DataLoader

from Datasets.coco import COCOKarpathyTest
from Datasets.flickr import FlickrTest
from Datasets.dataset_utils import getTestTransforms
from Datasets.tasviret import TasvirEtTest
from Model import TRCaptionNetpp

from utils import over_write_args


@torch.no_grad()
def predict(model, data_loader, device):
    # evaluate
    model.eval()
    result = []

    counter = 0
    for image, img_ids in data_loader:
        image = image.to(device)
        preds = model.generate(image)
        for pred, img_id in zip(preds, img_ids):
            result.append({"image_id": int(img_id), "caption": pred})
            counter += 1
    return result


def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    result = cocoEval.eval
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

    test_result = predict(model, test_loader, opt.device)

    os.makedirs(opt.output_dir, exist_ok=True)
    result_file = os.path.join(opt.output_dir, opt.prediction_file)
    json.dump(test_result, open(result_file, 'w'))

    result = evaluate_on_coco_caption(result_file,
                                      opt.test_json,
                                      os.path.join(opt.output_dir, opt.result_file))
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
