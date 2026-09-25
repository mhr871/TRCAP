"""Decisive test: is the decoder actually conditioning on the image?

Stage 2 training of the MobileCLIP hybrid produced the exact same degenerate
caption ("bir bir bir ...") for every single sample image, with metrics flat
across two eval checkpoints. That symptom is consistent with two very
different explanations that look identical in a training log:

  (a) a real wiring bug: the image signal never reaches (or is discarded by)
      the decoder, so generation is unconditional regardless of input, or
  (b) a training-dynamics problem: the freshly-initialized cross-attention
      layers haven't learned to use the image yet (mode collapse to the
      decoder's unconditional language-model prior), which can still resolve
      with more training.

Code review alone cannot distinguish these -- both predict the same log
output. This script settles it empirically against an already-trained
checkpoint (no need to wait for more training iterations):

  1. Loads the model + a checkpoint (e.g. the Stage 2 model_last.pth you
     already have).
  2. Picks several genuinely different test images.
  3. Confirms the vision encoder + projection actually produce different
     image_embeds per image (rules out an encoder/projection collapse).
  4. Runs model.generate() on each image individually and prints the result.
     If two very different images produce the exact same caption, that is
     decisive evidence of (a), not (b).
  5. Runs one training-style forward+backward pass and reports the gradient
     norm of the cross-attention parameters specifically (not the whole
     decoder, which is dominated by the pretrained self-attention/FFN
     layers that get gradient regardless of image conditioning) -- a
     vanishingly small cross-attention gradient relative to the rest of the
     decoder would support explanation (b).
"""

import argparse
import json
import random
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
    parser.add_argument("--num-images", type=int, default=5)
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
    with open(data_dir / "tasvir_test.json", "r", encoding="utf-8") as fp:
        images_meta = json.load(fp)["images"]

    random.seed(args.seed)
    picked = random.sample(images_meta, min(args.num_images, len(images_meta)))
    transform = getTestTransforms(model_config=config["model"])

    print("\n=== Step 1: do different images produce different encoder+proj features? ===")
    tensors = []
    for item in picked:
        path = image_path(images_root, item["filename"])
        tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        tensors.append((item["filename"], tensor))

    with torch.no_grad():
        embeds = []
        for filename, tensor in tensors:
            image_embeds = model.vision_encoder(tensor).float()
            if model.proj is not None:
                image_embeds = model.proj(image_embeds)
            embeds.append(image_embeds)
            print(f"  {filename}: image_embeds mean={image_embeds.mean().item():.6f}, "
                 f"std={image_embeds.std().item():.6f}")

    max_pairwise_diff = 0.0
    for i in range(len(embeds)):
        for j in range(i + 1, len(embeds)):
            diff = (embeds[i] - embeds[j]).abs().mean().item()
            max_pairwise_diff = max(max_pairwise_diff, diff)
    print(f"  -> largest pairwise mean|diff| between any two images' projected "
         f"features: {max_pairwise_diff:.6f}")
    if max_pairwise_diff < 1e-4:
        print("  [FINDING] Projected image features are essentially IDENTICAL across "
             "different images. The bug is upstream of the decoder (encoder or "
             "projection collapsed to a near-constant output).")
    else:
        print("  [OK] Projected image features clearly differ across images -- "
             "the encoder/projection are not the problem.")

    print("\n=== Step 2: does model.generate() actually produce different captions per image? ===")
    captions = []
    with torch.no_grad():
        for filename, tensor in tensors:
            caption = model.generate(tensor)[0]
            captions.append(caption)
            print(f"  {filename}: {caption!r}")

    unique_captions = set(captions)
    print(f"  -> {len(unique_captions)} unique caption(s) out of {len(captions)} images")
    if len(unique_captions) == 1:
        print("  [FINDING] Every image produced the EXACT SAME caption. Combined with "
             "Step 1's result, this tells us whether the bug is upstream (encoder/proj) "
             "or in the decoder/generation path itself.")
    else:
        print("  [OK] Different images produce different captions -- generation IS "
             "conditioning on the image at some level; the quality issue is a "
             "training-dynamics problem, not a wiring bug.")

    print("\n=== Step 3: is the cross-attention specifically receiving gradient? ===")
    model.train()
    model.zero_grad(set_to_none=True)
    dummy_tensor = tensors[0][1]
    dummy_caption = ["bu bir fotograf ."]
    loss = model(dummy_tensor, dummy_caption)
    loss.backward()

    cross_attn_grad_norm = 0.0
    cross_attn_param_count = 0
    other_decoder_grad_norm = 0.0
    other_decoder_param_count = 0
    for name, p in model.language_decoder.named_parameters():
        if p.grad is None:
            continue
        if "crossattention" in name:
            cross_attn_grad_norm += p.grad.norm().item() ** 2
            cross_attn_param_count += p.numel()
        else:
            other_decoder_grad_norm += p.grad.norm().item() ** 2
            other_decoder_param_count += p.numel()
    cross_attn_grad_norm **= 0.5
    other_decoder_grad_norm **= 0.5
    print(f"  cross-attention grad norm: {cross_attn_grad_norm:.6f} "
         f"({cross_attn_param_count:,} params)")
    print(f"  rest-of-decoder grad norm: {other_decoder_grad_norm:.6f} "
         f"({other_decoder_param_count:,} params)")
    if cross_attn_grad_norm == 0.0:
        print("  [FINDING] Cross-attention received ZERO gradient. This is a wiring bug.")
    else:
        print("  [OK] Cross-attention is receiving gradient. Whether it's ENOUGH after "
             "8000 steps at lr=1e-5 is a separate, training-dynamics question.")
    model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
