"""Same check as tools/check_pairing.py, but renders the actual images
inline (matplotlib grid) instead of only printing text -- for visually
spot-checking, in a Colab cell, whether the generated caption really
matches what is in the photo.

Run in a Colab cell (not `!python ...`, so the plot renders inline):

    %run tools/visualize_pairing.py --config configs/tasviret/mobileclip_s0_stage2.yaml \
        --weights /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_stage2_tasviret/model_last.pth

Or from Python directly:

    import sys; sys.argv = ["visualize_pairing.py", "--config", "...", "--weights", "..."]
    exec(open("tools/visualize_pairing.py").read())
"""

import argparse
import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
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


def wrap(text, width=42):
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="e.g. configs/tasviret/mobileclip_s0_stage2.yaml")
    parser.add_argument("--weights", required=True, help="checkpoint to load, e.g. model_last.pth")
    parser.add_argument("--data-dir", default="Data/tasvir-et")
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="Use the same seed as check_pairing.py to compare the same images.")
    parser.add_argument("--save", default=None, help="Optional path to also save the figure as a PNG.")
    args, _ = parser.parse_known_args()

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
    json_path = data_dir / f"tasvir_{args.split}.json"
    with open(json_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    images_meta = payload["images"]
    refs_by_image_id = defaultdict(list)
    for ann in payload.get("annotations", []):
        refs_by_image_id[ann["image_id"]].append(ann["caption"])

    random.seed(args.seed)
    picked = random.sample(images_meta, min(args.num_images, len(images_meta)))
    transform = getTestTransforms(model_config=config["model"])

    rows = (len(picked) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 4.2, rows * 5.4))
    axes = axes.flatten() if len(picked) > 1 else [axes]

    for ax, item in zip(axes, picked):
        img_id = item.get("imgid", item.get("id"))
        filename = item.get("filename", item.get("file_name"))
        path = image_path(images_root, filename)
        refs = refs_by_image_id.get(img_id, [])

        if path.exists():
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                generated = model.generate(tensor)[0]
            ax.imshow(image)
        else:
            generated = "[GÖRSEL BULUNAMADI]"
            ax.text(0.5, 0.5, "missing file", ha="center", va="center")

        ref_text = " / ".join(refs) if refs else "(referans yok)"
        title = f"REF: {wrap(ref_text)}\n\nÜRETİLEN: {wrap(generated)}"
        ax.set_title(title, fontsize=8, loc="left")
        ax.axis("off")

    for ax in axes[len(picked):]:
        ax.axis("off")

    fig.suptitle(f"{args.weights}  (split={args.split}, seed={args.seed})", fontsize=10)
    fig.tight_layout()
    if args.save:
        save_path = repo_path(args.save)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[OK] Saved figure to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
