"""Run the repo owner's own, untouched TRCaptionNetpp inference code
(copied byte-for-byte from ../TRCaptionNetpp/Model into official_trcaptionnetpp/
Model, isolated under its own `Model` namespace so it cannot accidentally
import anything from our own, modified Model/ package) against the TasvirEt
test set, and score it with the exact same evaluate_on_coco_caption() used
by trainer.py/eval.py everywhere else.

Purpose: our own Model/TRCaptionNet.py, Model/bert/med.py and Model/dino/dino.py
have all been touched by bug fixes over time (hardcoded CLS token id,
KV-cache prepare_inputs_for_generation rename, mobileclip support, ...).
A line-by-line diff against the pristine files already showed the
Turkish/DINOv2 generate() path is functionally identical (the only real
differences are a no-op-for-Turkish CLS-id fix and a KV-cache speed fix that
does not change output tokens), but this script gives an *empirical*,
undeniable confirmation of that: it generates captions with the literal
original code and feeds them through our metric pipeline. If this Bleu_4
matches (or is very close to) what eval.py already reported for the same
checkpoint, that proves the low score is not caused by our TRCaptionNet.py/
generate() implementation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = REPO_ROOT / "official_trcaptionnetpp"

# OFFICIAL_ROOT must win the "import Model" race: inserted at position 0 so
# `from Model import TRCaptionNetpp` (and everything TRCaptionNet.py itself
# imports, e.g. `from Model.bert import ...`) resolves against the pristine
# copy, never against the repo's own (modified) Model/ package.
sys.path.insert(0, str(OFFICIAL_ROOT))
from Model import TRCaptionNetpp  # noqa: E402  (pristine, from official_trcaptionnetpp/Model)

sys.path.insert(0, str(REPO_ROOT))
from Datasets.tasviret import TasvirEtTest  # noqa: E402
from eval import evaluate_on_coco_caption  # noqa: E402  (metric code only, touches no Model.*)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tasviret/tasviretpp_large_tasviret.yaml",
                        help="Only the 'model' section is read, purely to avoid re-typing the "
                             "architecture dict that is already identical to TRCaptionNetpp/app.py's.")
    parser.add_argument("--weights", default="checkpoints/TRCaptionNetpp_Large.pth")
    parser.add_argument("--test-json", default="Data/tasvir-et/tasvir_test.json")
    parser.add_argument("--test-data", default="Data/flickr8k/images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="eval_outputs/tasviret_test_official_trcaptionnetpp")
    args = parser.parse_args()

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as fp:
        model_config = yaml.safe_load(fp)["model"]

    print(f"[official_trcaptionnetpp] loading pristine Model.TRCaptionNetpp from {OFFICIAL_ROOT}")
    model = TRCaptionNetpp(model_config)
    checkpoint = torch.load(REPO_ROOT / args.weights, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    model.eval()
    print(f"[OK] pristine checkpoint loaded strict=True onto {args.device}")

    # Identical to TRCaptionNetpp/app.py's own `preprocess` (and to our
    # Datasets.dataset_utils.getDino2Transforms) -- Resize((224,224)) +
    # ToTensor + ImageNet mean/std normalization.
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = TasvirEtTest(dataset_root=str(REPO_ROOT / args.test_data),
                                json_path=str(REPO_ROOT / args.test_json),
                                transforms=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True, shuffle=False)

    results = []
    with torch.no_grad():
        for image, img_ids in test_loader:
            image = image.to(args.device)
            # No overrides: uses TRCaptionNetpp.generate()'s own defaults
            # (min_length=12, num_beams=3, repetition_penalty=1.1), the same
            # defaults eval.py's predict() relies on.
            preds = model.generate(image)
            for pred, img_id in zip(preds, img_ids):
                results.append({"image_id": int(img_id), "caption": pred})

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False)
    print(f"[OK] wrote {len(results)} captions to {predictions_path}")

    for sample in results[:5]:
        print("sample:", sample)

    metrics_path = output_dir / "metrics.json"
    result = evaluate_on_coco_caption(str(predictions_path), str(REPO_ROOT / args.test_json), str(metrics_path))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
