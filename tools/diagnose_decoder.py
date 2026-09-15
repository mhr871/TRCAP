"""Component-level test #3 of 3: does the DECODER actually use the visual
signal it's given, or does it mostly ignore it and fall back on language-
model habits learned from the training captions (e.g. the recurring
"X ve onu Y eden X" pattern seen in real generations)?

This needs a REAL TRAINED checkpoint (proj + decoder weights after actual
training) -- tools/diagnose_encoder.py and tools/diagnose_proj.py could run
locally with fresh/untrained weights, but this test is specifically about
what training produced, so it only makes sense against something like
mobileclip_s0_stage2_lr_exp1_tasviret/model_last.pth from Google Drive.

Method -- SWAP TEST, using the same 5 real, visually distinct sample images
as component tests #1-2 (images/test1.png .. test9.jpg):

  1. Compute each image's real proj-level embedding the normal way
     (vision_encoder -> proj), then generate its caption normally. This is
     the "own embedding" baseline.
  2. For each image, generate a SECOND caption but feed the decoder a
     DIFFERENT image's embedding instead (e.g. the cat's embedding is
     replaced with the ferry boat's). This bypasses model.generate()'s
     public API (which always recomputes embeddings from pixels) with a
     small helper that calls language_decoder.generate() directly, so the
     swapped-in embedding is used exactly the way a real one would be.

  Expected if the decoder is properly conditioned on the image: the
  "swapped" caption should read like it's about the OTHER image (e.g. the
  cat slot now mentions a boat/water), clearly different from the "own"
  caption.

  Finding if the decoder is NOT really using the image: "own" and
  "swapped" captions stay similar/generic regardless of which embedding
  was fed in -- concrete evidence the language-model prior dominates over
  the visual signal.

A simple word-overlap score between "own" and "swapped" captions per image
is printed as a rough, at-a-glance indicator alongside the actual text.
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Same 5 images as diagnose_encoder.py / diagnose_proj.py, for direct comparability.
SAMPLE_IMAGES = [
    ("images/test1.png", "kalp seklinde eller"),
    ("images/test4.png", "vapur"),
    ("images/test6.png", "kedi"),
    ("images/test7.png", "silo"),
    ("images/test9.jpg", "bayrak asan adam"),
]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def word_overlap(a, b):
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@torch.no_grad()
def generate_from_embeds(model, image_embeds, min_length=12, num_beams=3, repetition_penalty=1.1):
    """Mirrors TRCaptionNetpp.generate() exactly, except it takes an
    already-computed (post-proj) image_embeds tensor instead of raw pixels
    -- lets us feed in a DIFFERENT image's embedding on purpose."""
    image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
    model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
    input_ids = torch.ones((image_embeds.shape[0], 1), device=image_embeds.device, dtype=torch.long)
    input_ids *= model.tokenizer.cls_token_id
    outputs = model.language_decoder.generate(
        input_ids=input_ids,
        max_length=model.max_length,
        min_length=min_length,
        num_beams=num_beams,
        eos_token_id=model.tokenizer.sep_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
        repetition_penalty=repetition_penalty,
        **model_kwargs,
    )
    return [model.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="e.g. configs/tasviret/mobileclip_s0_stage2_lr_exp1.yaml")
    parser.add_argument("--weights", required=True, help="REAL TRAINED checkpoint, e.g. Drive's model_last.pth")
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
    print(f"[OK] Loaded REAL trained checkpoint {args.weights} onto {device}")

    transform = getTestTransforms(model_config=config["model"])

    names = []
    labels = []
    embeds = []
    own_captions = []
    with torch.no_grad():
        for rel_path, label in SAMPLE_IMAGES:
            img = Image.open(repo_path(rel_path)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            image_embeds = model.vision_encoder(tensor).float()
            if model.proj is not None:
                image_embeds = model.proj(image_embeds)
            embeds.append(image_embeds)
            names.append(Path(rel_path).name)
            labels.append(label)
            caption = generate_from_embeds(model, image_embeds)[0]
            own_captions.append(caption)

    print("\n=== 'Kendi' embedding'iyle uretilen caption'lar (normal caption uretimiyle ayni) ===")
    for name, label, cap in zip(names, labels, own_captions):
        print(f"  {name:16s} ({label:20s}): {cap!r}")

    print("\n=== TAKAS TESTI: her goruntuye BASKA bir goruntunun embedding'ini veriyoruz ===")
    print("(beklenti: caption artik takas edilen goruntunun icerigine benzer bir seye kaymali)\n")
    n = len(embeds)
    overlaps = []
    for i in range(n):
        j = (i + 1) % n  # swap with the "next" image, cyclic
        swapped_caption = generate_from_embeds(model, embeds[j])[0]
        overlap = word_overlap(own_captions[i], swapped_caption)
        overlaps.append(overlap)
        print(f"  {names[i]:16s} ({labels[i]:20s}) yerine -> {names[j]} ({labels[j]}) embedding'i verildi:")
        print(f"    kendi caption'i   : {own_captions[i]!r}")
        print(f"    takas caption'i   : {swapped_caption!r}")
        print(f"    kelime ortusmesi  : {overlap:.2f}  (0=tamamen farkli, 1=ozdes)")
        print()

    mean_overlap = sum(overlaps) / len(overlaps)
    print(f"=== ORTALAMA kelime ortusmesi (kendi vs takas): {mean_overlap:.2f} ===")
    if mean_overlap > 0.5:
        print("[FINDING] 'Kendi' ve 'takas' caption'lar cok benzer kaliyor (yuksek ortusme) -- "
             "decoder, hangi goruntunun embedding'i verildiginden pek etkilenmiyor gibi "
             "gorunuyor. Bu, gorsel sinyalin decoder tarafindan yeterince kullanilmadiginin "
             "somut bir kaniti olabilir.")
    else:
        print("[OK] 'Kendi' ve 'takas' caption'lar acikca farkliliyor -- decoder gercekten "
             "hangi embedding'in verildigine gore farkli cumleler kuruyor, yani gorsel "
             "sinyali kullaniyor.")


if __name__ == "__main__":
    main()
