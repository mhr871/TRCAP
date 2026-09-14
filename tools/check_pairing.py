"""Decisive test: is ground-truth reference text actually about the same
image it's paired with, and does the model's caption resemble it?

A trained model producing fluent, diverse-but-content-wrong captions (e.g.
"kayak yapan insanlar" for an image of two dogs) is consistent with at
least three very different root causes that look identical from a training
log or from diagnose_conditioning.py alone:

  (a) a data bug: the reference caption shown for image X was actually
      written for a different image (an upstream indexing/pairing bug in
      how tasvir_*.json was built or is read), so the model is being
      trained against the wrong target and no amount of training fixes it.
  (b) a genuine training/capacity issue: pairing is correct, but the model
      (only partially trained / a small mobile encoder) has not learned
      to extract the right content yet.
  (c) a manual-testing mistake: the script/checkpoint/config used for a
      one-off manual check did not match what was actually trained.

This script prints, for several real test images: the resolved image file
path (so you can visually open it), every ground-truth reference caption
for that exact image_id, and the model's generated caption -- side by
side, so a human can immediately tell (a) from (b) from (c).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def image_path(images_root, filename):
    candidates = [filename]
    if "_" in filename:
        candidates.append(f"{filename.split('_')[0]}.jpg")
    for candidate in candidates:
        path = images_root / candidate
        if path.exists():
            return path
    return images_root / filename


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="e.g. configs/tasviret/mobileclip_s0_stage2.yaml")
    parser.add_argument("--weights", required=True, help="checkpoint to load, e.g. model_last.pth")
    parser.add_argument("--data-dir", default="Data/tasvir-et")
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--json-prefix", default="tasvir",
                        help="'tasvir' (default, TasvirEt) or 'flickr8k' for the English "
                             "Karpathy split from tools/prepare_flickr8k_en.py "
                             "(use with --data-dir Data/flickr8k-en).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from Datasets.dataset_utils import getTestTransforms
    from Model import TRCaptionNetpp

    config_path = repo_path(args.config)
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TRCaptionNetpp(config["model"])
    checkpoint = torch.load(repo_path(args.weights), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"[OK] Loaded {args.weights} onto {device}")

    data_dir = repo_path(args.data_dir)
    images_root = repo_path(args.images_root)
    json_path = data_dir / f"{args.json_prefix}_{args.split}.json"
    with open(json_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    images_meta = payload["images"]
    refs_by_image_id = defaultdict(list)
    for ann in payload.get("annotations", []):
        refs_by_image_id[ann["image_id"]].append(ann["caption"])

    random.seed(args.seed)
    picked = random.sample(images_meta, min(args.num_images, len(images_meta)))
    transform = getTestTransforms(model_config=config["model"])

    print(f"\nUsing {json_path}, {len(images_meta)} images, "
         f"{sum(len(v) for v in refs_by_image_id.values())} reference captions total.\n")

    for item in picked:
        img_id = item.get("imgid", item.get("id"))
        filename = item.get("filename", item.get("file_name"))
        path = image_path(images_root, filename)
        refs = refs_by_image_id.get(img_id, [])

        print(f"=== image_id={img_id}  file={filename} ===")
        print(f"  resolved path: {path}  (exists={path.exists()})")
        if not refs:
            print("  [WARNING] no reference captions found for this image_id -- "
                 "check that annotations[].image_id actually matches images[].id/imgid.")
        for i, ref in enumerate(refs, start=1):
            print(f"  reference {i}: {ref!r}")

        if path.exists():
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                generated = model.generate(tensor)[0]
            print(f"  generated   : {generated!r}")
        else:
            print("  [SKIPPED generation] image file missing")
        print()


if __name__ == "__main__":
    main()
