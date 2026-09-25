"""Component-level test #1 of 3: is the MobileCLIP VISION ENCODER, on its
own, actually a sound, meaningful image feature extractor -- independent of
proj/decoder? Two separate sub-checks, because they test two different
things:

Part A -- is the pretrained checkpoint itself good?
    Uses the official `mobileclip` package's OWN image+text encoders and
    its OWN native usage (mobileclip.create_model_and_transforms +
    mobileclip.get_tokenizer), exactly as apple/ml-mobileclip intends. This
    is the model's real CLIP joint embedding space (pooled), so cosine
    similarity against short ENGLISH text prompts is meaningful (MobileCLIP
    was pretrained on English image-text pairs, not Turkish -- Turkish
    prompts would not be a fair test of the checkpoint). If the correct
    image/prompt pairs score clearly higher than the wrong pairs, the
    downloaded checkpoint is a genuinely working CLIP model.

Part B -- does OUR wrapper (Model/mobileclip/mobileclip_encoder.py) produce
    a usable signal?
    Our wrapper does NOT use the pooled CLIP embedding -- it returns the
    raw pre-pooling conv_exp feature map (patch tokens), because that's
    what the cross-attention decoder conditions on. That output is not
    text-aligned, so it can't be scored against text. Instead this checks
    the more basic property: do different images actually produce
    different raw feature maps (not a collapsed/near-constant output)?

Run this before looking at proj or decoder -- if either part fails here,
nothing built on top of the encoder can be trusted.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 5 visually and semantically distinct sample images already committed in
# this repo under images/ (no TasvirEt download needed) with short, honest
# ENGLISH descriptions for Part A.
SAMPLES = [
    ("images/test1.png", "two hands forming a heart shape over the sea"),
    ("images/test4.png", "a passenger ferry boat on the water"),
    ("images/test6.png", "a cat sitting outdoors"),
    ("images/test7.png", "industrial grain storage silos"),
    ("images/test9.jpg", "a man hanging a flag on a balcony"),
]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def part_a_checkpoint_soundness(weights_path, model_name, device):
    import mobileclip

    print("\n" + "=" * 70)
    print("PART A: is the pretrained checkpoint itself a sound CLIP model?")
    print("(official mobileclip package, native usage, English prompts)")
    print("=" * 70)

    model, _, preprocess = mobileclip.create_model_and_transforms(model_name, pretrained=str(weights_path))
    tokenizer = mobileclip.get_tokenizer(model_name)
    model = model.to(device).eval()

    # Model/mobileclip/mobileclip_encoder.py claims (in a comment, never
    # re-verified against the real package) that MobileCLIP uses IDENTITY
    # normalization (mean=(0,0,0), std=(1,1,1)) instead of CLIP-style
    # ImageNet mean/std, and that Datasets.dataset_utils.getMobileCLIPTransforms
    # matches this. Print the package's own official `preprocess` here so
    # that claim can be checked against reality instead of trusted blindly.
    print("\n--- sanity: official mobileclip preprocess vs. our own constants ---")
    print(f"  official preprocess (from mobileclip.create_model_and_transforms):\n    {preprocess}")
    from Model.mobileclip import MOBILECLIP_MEAN, MOBILECLIP_STD, MOBILECLIP_IMAGE_SIZE  # noqa: E402
    print(f"  our own constants: MOBILECLIP_IMAGE_SIZE={MOBILECLIP_IMAGE_SIZE}, "
         f"MOBILECLIP_MEAN={MOBILECLIP_MEAN}, MOBILECLIP_STD={MOBILECLIP_STD}")
    print("  -> compare the Normalize(mean=..., std=...) and Resize/CenterCrop size "
         "printed above by hand: if they don't match what our own transform uses "
         "in training/eval, that is a real preprocessing mismatch bug.")

    images = []
    prompts = []
    for rel_path, prompt in SAMPLES:
        img = Image.open(repo_path(rel_path)).convert("RGB")
        images.append(preprocess(img).unsqueeze(0))
        prompts.append(prompt)
    image_batch = torch.cat(images, dim=0).to(device)
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).cpu()

    names = [Path(p).name for p, _ in SAMPLES]
    header = " " * 22 + "".join(f"{n:>14s}" for n in names)
    print("\ncosine similarity matrix (rows=image, cols=text prompt):")
    print(header)
    for i, name in enumerate(names):
        row = "".join(f"{similarity[i, j].item():14.3f}" for j in range(len(names)))
        print(f"{name:22s}{row}")

    diag = similarity.diag()
    off_diag_max = similarity.clone()
    off_diag_max.fill_diagonal_(float("-inf"))
    off_diag_max_per_row = off_diag_max.max(dim=1).values

    print("\nper-image: correct-prompt score vs. best wrong-prompt score")
    all_correct_win = True
    for i, name in enumerate(names):
        correct = diag[i].item()
        best_wrong = off_diag_max_per_row[i].item()
        win = correct > best_wrong
        all_correct_win &= win
        print(f"  {name:22s} correct={correct:.3f}  best_wrong={best_wrong:.3f}  {'OK' if win else '[FAIL]'}")

    if all_correct_win:
        print("\n[OK] Every image's correct prompt outscored every wrong prompt. "
             "The downloaded checkpoint is a working CLIP model.")
    else:
        print("\n[FINDING] At least one image did NOT match its own prompt best. "
             "The checkpoint itself may be corrupted, mismatched, or "
             "downloaded incorrectly -- worth re-checking before looking "
             "at proj or decoder at all.")


def part_b_wrapper_differentiation(weights_path, model_name, device):
    from Model.mobileclip import MobileCLIPEncoder, MOBILECLIP_MEAN, MOBILECLIP_STD, MOBILECLIP_IMAGE_SIZE
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    print("\n" + "=" * 70)
    print("PART B: does OUR wrapper's raw (pre-pool) feature map differ per image?")
    print("(Model/mobileclip/mobileclip_encoder.py, as TRCaptionNet.py actually uses it)")
    print("=" * 70)

    encoder = MobileCLIPEncoder(model_name, checkpoint_path=str(weights_path)).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(MOBILECLIP_IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(MOBILECLIP_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MOBILECLIP_MEAN, std=MOBILECLIP_STD),
    ])

    feats = []
    names = []
    with torch.no_grad():
        for rel_path, _ in SAMPLES:
            img = Image.open(repo_path(rel_path)).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            feat = encoder(tensor).float()
            feats.append(feat)
            names.append(Path(rel_path).name)
            print(f"  {Path(rel_path).name:22s} shape={tuple(feat.shape)}  "
                 f"mean={feat.mean().item():10.4f}  std={feat.std().item():10.4f}")

    print("\npairwise mean absolute difference between images' raw feature maps:")
    header = " " * 22 + "".join(f"{n:>14s}" for n in names)
    print(header)
    max_diff = 0.0
    min_diff = float("inf")
    for i in range(len(feats)):
        row_vals = []
        for j in range(len(feats)):
            if i == j:
                row_vals.append(0.0)
                continue
            d = (feats[i] - feats[j]).abs().mean().item()
            row_vals.append(d)
            if i != j:
                max_diff = max(max_diff, d)
                min_diff = min(min_diff, d)
        print(f"{names[i]:22s}" + "".join(f"{v:14.4f}" for v in row_vals))

    print(f"\nsmallest pairwise diff (any two different images): {min_diff:.6f}")
    print(f"largest pairwise diff  (any two different images): {max_diff:.6f}")
    if min_diff < 1e-4:
        print("[FINDING] At least two visually very different images produced "
             "nearly IDENTICAL raw features. The encoder wrapper output is "
             "collapsing -- a real upstream bug.")
    else:
        print("[OK] All 5 visually distinct images produce clearly different raw "
             "feature maps. The encoder wrapper itself is not collapsed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mobileclip_s0")
    parser.add_argument("--weights", default="checkpoints/mobileclip_s0.pt")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = repo_path(args.weights)
    print(f"[OK] using device={device}, weights={weights_path}")

    part_a_checkpoint_soundness(weights_path, args.model, device)
    part_b_wrapper_differentiation(weights_path, args.model, device)


if __name__ == "__main__":
    main()
