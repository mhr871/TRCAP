"""Component-level test #2 of 3: does the projection layer (MlpProj:
LayerNorm -> Linear -> GELU -> Linear) that bridges MobileCLIP's raw patch
features to the decoder actually work correctly, or does it corrupt the
signal on the way through?

Context: Model/TRCaptionNet.py's MlpProj docstring claims raw MobileCLIP
features have a wildly inconsistent per-image scale ("std ranged
22198-134717 across 5 sample images"), citing tools/diagnose_conditioning.py
as the source. Re-reading that script shows it actually measures features
AFTER proj is applied (`image_embeds = model.proj(image_embeds)` happens
BEFORE the printed stats), not the raw encoder output -- so that historical
number was mislabeled. tools/diagnose_encoder.py (component test #1) just
measured the true raw encoder output on 5 real images and found a small,
sane std (~0.19-0.27), nothing like 22198-134717. So either the LayerNorm
fix was solving a problem that was actually in the *old* proj
implementation (not the raw features as documented), or something else
entirely.

This script measures, on the SAME 5 images as diagnose_encoder.py, with a
freshly-initialized (untrained) MlpProj -- so it tests the architecture
itself, independent of any specific training run:

  1. Scale consistency: does proj output have a similar std/norm across
     different images (good), or does it vary wildly per image (bad --
     the decoder would receive a signal whose sheer magnitude swings
     around for reasons unrelated to content)?
  2. Differentiation preserved: do different images still produce clearly
     different proj outputs (good), or does proj collapse them toward a
     near-constant vector (bad -- the decoder would receive almost no
     usable per-image information at all)?

Run after tools/diagnose_encoder.py (component test #1). A real trained
checkpoint's proj weights can be checked the same way later by pointing
this script at real weights instead of a fresh random init.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Same 5 images as tools/diagnose_encoder.py, for direct comparability.
SAMPLE_IMAGES = [
    "images/test1.png",
    "images/test4.png",
    "images/test6.png",
    "images/test7.png",
    "images/test9.jpg",
]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mobileclip_s0")
    parser.add_argument("--weights", default="checkpoints/mobileclip_s0.pt",
                        help="MobileCLIP encoder checkpoint. Ignored if --trained-weights is given "
                             "(the full checkpoint's own encoder+proj are used instead).")
    parser.add_argument("--seed", type=int, default=0, help="proj weight init seed (fresh-proj mode only)")
    parser.add_argument("--trained-config", default=None,
                        help="If given together with --trained-weights: load the FULL trained "
                             "TRCaptionNetpp model (e.g. configs/tasviret/mobileclip_s0_stage2_lr_exp1.yaml) "
                             "and use its REAL, trained vision_encoder+proj instead of a fresh one.")
    parser.add_argument("--trained-weights", default=None,
                        help="REAL trained checkpoint, e.g. Drive's model_last.pth. Requires --trained-config.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[OK] using device={device}")

    if args.trained_weights:
        if not args.trained_config:
            raise SystemExit("--trained-weights requires --trained-config")
        import yaml
        from Model import TRCaptionNetpp

        with open(repo_path(args.trained_config), "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)["model"]
        model = TRCaptionNetpp(config)
        checkpoint = torch.load(repo_path(args.trained_weights), map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()
        encoder = model.vision_encoder
        proj = model.proj
        print(f"[OK] loaded REAL TRAINED checkpoint {args.trained_weights} "
             f"(config={args.trained_config}) -- using its actual trained proj")
    else:
        from Model.mobileclip import MobileCLIPEncoder
        from Model.TRCaptionNet import MlpProj

        encoder = MobileCLIPEncoder(args.model, checkpoint_path=str(repo_path(args.weights))).to(device).eval()
        encoder_output_size = encoder.get_output_dim()
        print(f"[OK] encoder loaded, output_dim={encoder_output_size}")

        torch.manual_seed(args.seed)
        proj = MlpProj(encoder_output_size).to(device).eval()
        print(f"[OK] fresh (untrained, seed={args.seed}) MlpProj created")

    from Model.mobileclip import MOBILECLIP_MEAN, MOBILECLIP_STD, MOBILECLIP_IMAGE_SIZE
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    preprocess = transforms.Compose([
        transforms.Resize(MOBILECLIP_IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(MOBILECLIP_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MOBILECLIP_MEAN, std=MOBILECLIP_STD),
    ])

    names = []
    raw_stats = []
    proj_stats = []
    proj_vecs = []
    with torch.no_grad():
        for rel_path in SAMPLE_IMAGES:
            img = Image.open(repo_path(rel_path)).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)

            raw = encoder(tensor).float()
            projected = proj(raw)

            names.append(Path(rel_path).name)
            raw_stats.append((raw.mean().item(), raw.std().item(), raw.norm(dim=-1).mean().item()))
            proj_stats.append((projected.mean().item(), projected.std().item(), projected.norm(dim=-1).mean().item()))
            proj_vecs.append(projected)

    print(f"\n{'image':16s} {'raw mean':>10s} {'raw std':>10s} {'raw norm':>10s}   "
         f"{'proj mean':>10s} {'proj std':>10s} {'proj norm':>10s}")
    for name, r, p in zip(names, raw_stats, proj_stats):
        print(f"{name:16s} {r[0]:10.4f} {r[1]:10.4f} {r[2]:10.4f}   {p[0]:10.4f} {p[1]:10.4f} {p[2]:10.4f}")

    def ratio(stats, idx):
        vals = [s[idx] for s in stats]
        lo, hi = min(vals), max(vals)
        return lo, hi, (hi / lo if lo > 0 else float("inf"))

    print("\n=== Olcum 1: OLCEK TUTARLILIGI (std'nin goruldenal goruntuye orani, kucuk=iyi) ===")
    for label, stats in [("raw std", raw_stats), ("proj std", proj_stats)]:
        lo, hi, r = ratio(stats, 1)
        verdict = "OK" if r < 3.0 else "[FINDING] tutarsiz olcek"
        print(f"  {label:10s}: min={lo:10.4f}  max={hi:10.4f}  oran(max/min)={r:8.2f}x   {verdict}")

    print("\n=== Olcum 2: AYIRT EDICILIK (5 goruntunun proj-sonrasi birbirinden farkli mi) ===")
    max_diff = 0.0
    min_diff = float("inf")
    for i in range(len(proj_vecs)):
        for j in range(i + 1, len(proj_vecs)):
            d = (proj_vecs[i] - proj_vecs[j]).abs().mean().item()
            max_diff = max(max_diff, d)
            min_diff = min(min_diff, d)
    print(f"  proj-sonrasi en kucuk ikili fark: {min_diff:.6f}")
    print(f"  proj-sonrasi en buyuk ikili fark: {max_diff:.6f}")
    if min_diff < 1e-4:
        print("  [FINDING] En az iki farkli goruntu proj SONRASI neredeyse ozdes vektore "
             "dusuyor -- proj bilgi kaybina/collapse'a yol aciyor olabilir.")
    else:
        print("  [OK] 5 goruntu proj sonrasi da birbirinden acikca farkli -- proj (taze "
             "haliyle) ayirt ediciligi koruyor.")

    print("\nNot: Bu olcum TAZE (egitilmemis) bir proj ile yapildi -- mimarinin kendisini "
         "test ediyor. Gercek egitilmis checkpoint (orn. Colab'daki stage2 model_last.pth) "
         "ile de tekrarlanmali; bu script sadece --weights degil proj agirliklarini da "
         "yukleyecek sekilde genisletilebilir.")


if __name__ == "__main__":
    main()
