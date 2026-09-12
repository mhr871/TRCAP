"""Preflight check for the MobileCLIP hybrid pipeline (S0/S1/S2 + BERTurk).

Unlike tools/preflight_colab.py (which pins exact hyperparameters/SHA256 for
the frozen DINOv2 baseline reproduction), this script validates the *new*
mobileclip_* configs: runtime, TasvirEt data integrity (same checks, dataset
is unchanged), the mobileclip package/checkpoint, and an actual forward +
backward + generate() smoke test so a real training run does not fail after
several minutes of setup.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import transformers
import yaml
from PIL import Image

EXPECTED_SPLITS = {
    "train": (6000, 12028),
    "val": (1000, 2006),
    "test": (1000, 2003),
}
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


def check_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Select a GPU runtime before training.")
    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / (1024 ** 3)
    print(f"[OK] Python {sys.version.split()[0]}")
    print(f"[OK] PyTorch {torch.__version__}, Transformers {transformers.__version__}")
    print(f"[OK] GPU {properties.name}, VRAM {vram_gib:.2f} GiB")
    # METEOR/SPICE are skipped in eval.py (Java subprocess dependency
    # removed after it crashed training on a malformed stats line), so Java
    # is no longer required here.


def check_data(data_dir, images_root):
    split_ids = {}
    all_image_paths = []
    for split, expected_counts in EXPECTED_SPLITS.items():
        path = data_dir / f"tasvir_{split}.json"
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        counts = (len(payload["images"]), len(payload["annotations"]))
        if counts != expected_counts:
            raise ValueError(f"{split} counts are {counts}, expected {expected_counts}")
        split_ids[split] = {int(item["id"]) for item in payload["images"]}
        for item in payload["images"]:
            resolved = image_path(images_root, item["filename"])
            if not resolved.exists() or resolved.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty image: {resolved}")
            all_image_paths.append(resolved)
        print(f"[OK] {split}: {counts[0]} images, {counts[1]} captions")

    if split_ids["train"] & split_ids["val"] or split_ids["train"] & split_ids["test"] or split_ids["val"] & split_ids["test"]:
        raise ValueError("Train/validation/test image ids overlap")
    if len(set.union(*split_ids.values())) != 8000:
        raise ValueError("Train/validation/test union does not contain exactly 8000 image ids")
    print(f"[OK] All {len(all_image_paths)} referenced image files are present and disjoint by split")


def check_mobileclip_package(model_name, checkpoint_path):
    try:
        import mobileclip  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "mobileclip package not importable. Install with:\n"
            "  pip install --no-deps git+https://github.com/apple/ml-mobileclip.git"
        ) from exc
    print(f"[OK] mobileclip package importable (module: {mobileclip.__file__})")

    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing/empty checkpoint: {checkpoint_path}. Run:\n"
            f"  python tools/download_mobileclip.py --model {model_name}"
        )
    print(f"[OK] Checkpoint present: {checkpoint_path} ({checkpoint_path.stat().st_size} bytes)")


def model_smoke_test(config, images_root, test_json):
    transformers.logging.set_verbosity_error()
    from Datasets.dataset_utils import getTestTransforms
    from Model import TRCaptionNetpp

    freeze_decoder = bool(config.get("freeze_decoder", False))
    model = TRCaptionNetpp(config["model"])
    device = torch.device("cuda:0")
    model = model.to(device)
    if freeze_decoder:
        for p in model.language_decoder.parameters():
            p.requires_grad_(False)
    print(f"[OK] Model constructed (mobileclip={config['model']['mobileclip']}, "
         f"proj_type={config['model'].get('proj_type')}, freeze_decoder={freeze_decoder})")

    with open(test_json, "r", encoding="utf-8") as fp:
        sample = json.load(fp)["images"][0]
    sample_path = image_path(images_root, sample["filename"])
    transform = getTestTransforms(model_config=config["model"])
    image = transform(Image.open(sample_path).convert("RGB")).unsqueeze(0).to(device)

    # forward + backward: confirms gradients reach the projection layer (and,
    # in stage 2, the decoder), and that the frozen encoder/decoder truly get
    # no gradient.
    model.train()
    if freeze_decoder:
        model.language_decoder.eval()
    # Placeholder text: the smoke test only checks that gradients flow
    # through the right modules, not caption quality, so the exact wording
    # doesn't matter (and tasvir_test.json's captions live in "annotations",
    # keyed by image_id, not inline on the image entry).
    caption = ["bu bir fotograf ."]
    loss = model(image, caption)
    loss.backward()

    proj_grad_norm = sum(p.grad.norm().item() for p in model.proj.parameters() if p.grad is not None)
    if proj_grad_norm == 0.0:
        raise RuntimeError("Projection layer received zero gradient; check proj wiring.")
    print(f"[OK] Forward+backward smoke test: loss={loss.item():.4f}, proj_grad_norm={proj_grad_norm:.6f}")

    encoder_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                           for p in model.vision_encoder.parameters())
    if encoder_has_grad:
        raise RuntimeError("Vision encoder received a gradient; it must stay frozen.")
    print("[OK] Vision encoder confirmed frozen (no gradient)")

    if freeze_decoder:
        decoder_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                               for p in model.language_decoder.parameters())
        if decoder_has_grad:
            raise RuntimeError("Language decoder received a gradient while freeze_decoder=true.")
        print("[OK] Language decoder confirmed frozen (no gradient) for stage 1")
    else:
        decoder_grad_norm = sum(p.grad.norm().item() for p in model.language_decoder.parameters() if p.grad is not None)
        if decoder_grad_norm == 0.0:
            raise RuntimeError("Language decoder received zero gradient with freeze_decoder=false; check wiring.")
        print(f"[OK] Language decoder receiving gradient as expected (norm={decoder_grad_norm:.6f})")

    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        generated = model.generate(image)[0]
    print(f"[OK] End-to-end GPU generation: {generated!r}")


def main():
    parser = argparse.ArgumentParser(description="Validate a MobileCLIP hybrid config before starting a real training run.")
    parser.add_argument("--config", required=True,
                        help="e.g. configs/tasviret/mobileclip_s0_stage1.yaml")
    parser.add_argument("--data-dir", default="Data/tasvir-et")
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--skip-model-smoke-test", action="store_true")
    args = parser.parse_args()

    config_path = repo_path(args.config)
    data_dir = repo_path(args.data_dir)
    images_root = repo_path(args.images_root)
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    model_name = config["model"]["mobileclip"]
    checkpoint_path = repo_path(config["model"]["mobileclip_ckpt"])

    check_runtime()
    check_data(data_dir, images_root)
    check_mobileclip_package(model_name, checkpoint_path)
    if not args.skip_model_smoke_test:
        model_smoke_test(config, images_root, data_dir / "tasvir_test.json")
    print(f"PREFLIGHT PASSED: {args.config} is ready for training.")


if __name__ == "__main__":
    main()
