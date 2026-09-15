"""Same qualitative check as tools/check_pairing.py (image path + every
ground-truth reference + generated caption, side by side, for a few real
test images) but using the repo owner's own, untouched TRCaptionNetpp code
(official_trcaptionnetpp/Model, copied byte-for-byte from ../TRCaptionNetpp)
instead of our own Model/ package.

Purpose: let a human directly compare, image by image, what "the guy's own
code" (repo owner's pristine demo implementation) generates versus what our
tools/check_pairing.py already showed for the same checkpoint. If the two
scripts print the same captions for the same images, that is a second,
qualitative confirmation (on top of eval_official_trcaptionnetpp.py's BLEU-4
number) that our generation code is not the source of the low baseline
score.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = REPO_ROOT / "official_trcaptionnetpp"

# Same isolation trick as eval_official_trcaptionnetpp.py: OFFICIAL_ROOT
# must win the "import Model" race, and nothing here may import anything
# that would drag in our own Model/ (e.g. Datasets.dataset_utils, which
# imports Model.mobileclip and would collide once "Model" is pinned to the
# pristine package). The image preprocessing is inlined below instead of
# imported for the same reason -- it's identical to
# Datasets.dataset_utils.getDino2Transforms()/TRCaptionNetpp/app.py's own
# `preprocess` anyway (Resize((224,224)) + ToTensor + ImageNet mean/std).
sys.path.insert(0, str(OFFICIAL_ROOT))
from Model import TRCaptionNetpp  # noqa: E402  (pristine, from official_trcaptionnetpp/Model)

sys.path.insert(0, str(REPO_ROOT))


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
    return images_root / candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tasviret/tasviretpp_large_tasviret.yaml")
    parser.add_argument("--weights", default="checkpoints/TRCaptionNetpp_Large.pth")
    parser.add_argument("--data-dir", default="Data/tasvir-et")
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
    print(f"[OK] Loaded {args.weights} onto {device} via pristine official_trcaptionnetpp/Model")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = repo_path(args.data_dir)
    images_root = repo_path(args.images_root)
    json_path = data_dir / f"tasvir_{args.split}.json"
    with open(json_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    images_meta = payload["images"]
    refs_by_image_id = defaultdict(list)
    for ann in payload.get("annotations", []):
        refs_by_image_id[ann["image_id"]].append(ann["caption"])

    random.seed(args.seed)
    picked = random.sample(images_meta, min(args.num_images, len(images_meta)))

    print(f"\nUsing {json_path}, {len(images_meta)} images, "
         f"{sum(len(v) for v in refs_by_image_id.values())} reference captions total.\n")

    for item in picked:
        img_id = item.get("imgid", item.get("id"))
        filename = item.get("filename", item.get("file_name"))
        path = image_path(images_root, filename)
        refs = refs_by_image_id.get(img_id, [])

        print(f"=== image_id={img_id}  file={filename} ===")
        print(f"  resolved path: {path}  (exists={path.exists()})")
        for i, ref in enumerate(refs, start=1):
            print(f"  reference {i}: {ref!r}")

        if path.exists():
            image = Image.open(path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                generated = model.generate(tensor)[0]
            print(f"  generated (official code): {generated!r}")
        else:
            print("  [SKIPPED generation] image file missing")
        print()


if __name__ == "__main__":
    main()
