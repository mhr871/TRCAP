"""Decisive test: is the vision-encoder signal actually stable enough for
the decoder to learn to use it, or is it still noisy/inconsistent the way
devam.md's MlpProj docstring describes (raw MobileCLIP features with
per-image-inconsistent scale, std ranging 22198-134717 across 5 images)?

That finding was used to justify adding a LayerNorm in front of the
2-layer MLP projection (Model/TRCaptionNet.py's MlpProj), but the fix was
never actually re-measured after being added -- it was reasoned about, not
verified. This script measures both sides directly, on real test images,
for a real trained checkpoint:

  1. RAW vision_encoder(image) output (before any proj): per-image mean,
     std, and L2 norm. Reproduces devam.md's original measurement to
     confirm it's still true (or not) for the current encoder/checkpoint.
  2. AFTER proj (LayerNorm -> Linear -> GELU -> Linear, i.e. what the
     decoder's cross-attention actually receives): same statistics. If
     LayerNorm is doing its job, these should be far more consistent
     across images than the raw ones -- similar mean/std regardless of
     image content, only *directionally* different (which is what carries
     information the decoder can use).

If step 2 is still wildly inconsistent in scale (not just direction)
across images, that is concrete evidence the decoder is receiving a noisy,
hard-to-use signal -- a plausible root cause for the "fluent but
content-generic / repetitive" captions, independent of lr.
"""

import argparse
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
    parser.add_argument("--config", required=True, help="e.g. configs/tasviret/mobileclip_s0_stage2_lr_exp1.yaml")
    parser.add_argument("--weights", required=True, help="checkpoint to load, e.g. model_last.pth")
    parser.add_argument("--data-dir", default="Data/tasvir-et")
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import json
    import random

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
    with open(data_dir / "tasvir_test.json", "r", encoding="utf-8") as fp:
        images_meta = json.load(fp)["images"]

    random.seed(args.seed)
    picked = random.sample(images_meta, min(args.num_images, len(images_meta)))
    transform = getTestTransforms(model_config=config["model"])

    raw_stats = []
    proj_stats = []
    print(f"\n{'filename':40s} {'raw mean':>12s} {'raw std':>12s} {'raw L2norm':>12s}   "
         f"{'proj mean':>12s} {'proj std':>12s} {'proj L2norm':>12s}")
    with torch.no_grad():
        for item in picked:
            filename = item.get("filename", item.get("file_name"))
            path = image_path(images_root, filename)
            if not path.exists():
                print(f"  [SKIP] missing image: {path}")
                continue
            tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

            raw = model.vision_encoder(tensor).float()
            raw_mean, raw_std = raw.mean().item(), raw.std().item()
            raw_norm = raw.norm(dim=-1).mean().item()
            raw_stats.append((raw_mean, raw_std, raw_norm))

            if model.proj is not None:
                projected = model.proj(raw)
                proj_mean, proj_std = projected.mean().item(), projected.std().item()
                proj_norm = projected.norm(dim=-1).mean().item()
            else:
                proj_mean = proj_std = proj_norm = float("nan")
            proj_stats.append((proj_mean, proj_std, proj_norm))

            print(f"{filename:40s} {raw_mean:12.3f} {raw_std:12.3f} {raw_norm:12.3f}   "
                 f"{proj_mean:12.4f} {proj_std:12.4f} {proj_norm:12.4f}")

    def spread(stats, idx):
        vals = [s[idx] for s in stats]
        return min(vals), max(vals), (max(vals) / min(vals)) if min(vals) != 0 else float("inf")

    print("\n=== Ozet: en kucuk / en buyuk / oran (max/min) ===")
    for name, idx in [("raw std", 1), ("raw L2norm", 2), ("proj std", 1), ("proj L2norm", 2)]:
        stats = raw_stats if name.startswith("raw") else proj_stats
        lo, hi, ratio = spread(stats, idx)
        print(f"  {name:12s}: min={lo:12.4f}  max={hi:12.4f}  oran(max/min)={ratio:8.2f}x")

    print("\nYorum: 'raw' satirlarindaki oran devam.md'nin bahsettigi olcek "
         "tutarsizligini yeniden olcuyor. 'proj' satirlarindaki oran, LayerNorm+MLP'nin "
         "bunu gercekten dengeleyip dengelemedigini gosteriyor -- oran hala buyukse "
         "(orn. >3-5x), decoder'a giden sinyal hala goruntuden goruntuye tutarsiz "
         "olcekte demektir.")


if __name__ == "__main__":
    main()
