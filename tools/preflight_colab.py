import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import torch
import transformers
import yaml
from PIL import Image


EXPECTED_CHECKPOINT_BYTES = 2438124871
EXPECTED_CHECKPOINT_SHA256 = "c055ef247f968c86140b941506026721ca4c301ef3c7f6b421caec89ada8ebf3"
EXPECTED_SPLITS = {
    "train": (6000, 12028),
    "val": (1000, 2006),
    "test": (1000, 2003),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(images_root, filename):
    candidates = [filename]
    if "_" in filename:
        candidates.append(f"{filename.split('_')[0]}.jpg")
    for candidate in candidates:
        path = images_root / candidate
        if path.exists():
            return path
    return images_root / filename


def check_config(config):
    expected = {
        "init_model_ckpt": "checkpoints/TRCaptionNetpp_Large.pth",
        "strict_init": True,
        "lr": 5e-4,
        "lr_proj": 5e-4,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "batch_size": 64,
        "max_iter": 50000,
        "warm_up_iter": 10000,
        "num_eval_iter": 8000,
        "target_metric": "Bleu_4",
    }
    for key, value in expected.items():
        actual = config.get(key)
        if key in {"lr", "lr_proj"}:
            actual = float(actual)
        if actual != value:
            raise ValueError(f"Config mismatch for {key}: expected {value!r}, got {actual!r}")

    expected_model = {
        "dino2": "dinov2_vitl14",
        "image_size": 224,
        "bert": "dbmdz/electra-base-turkish-mc4-cased-discriminator",
        "max_length": 35,
        "proj": True,
        "proj_num_head": 16,
    }
    for key, value in expected_model.items():
        if config["model"].get(key) != value:
            raise ValueError(
                f"Model config mismatch for {key}: expected {value!r}, got {config['model'].get(key)!r}"
            )
    print("[OK] Baseline architecture and training hyperparameters")


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

    for index, path in enumerate(all_image_paths, start=1):
        with Image.open(path) as image:
            image.verify()
        if index % 1000 == 0:
            print(f"Verified {index}/8000 image files")
    print("[OK] All 8000 image files are present, disjoint by split, and decodable")


def check_checkpoint(checkpoint_path):
    size = checkpoint_path.stat().st_size
    if size != EXPECTED_CHECKPOINT_BYTES:
        raise ValueError(f"Checkpoint size is {size}, expected {EXPECTED_CHECKPOINT_BYTES}")
    digest = sha256(checkpoint_path)
    if digest.lower() != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA256 is {digest}, expected {EXPECTED_CHECKPOINT_SHA256}")
    print(f"[OK] Checkpoint bytes and SHA256: {digest}")


def check_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Select a GPU runtime before training.")
    properties = torch.cuda.get_device_properties(0)
    vram_gib = properties.total_memory / (1024 ** 3)
    print(f"[OK] Python {sys.version.split()[0]}")
    print(f"[OK] PyTorch {torch.__version__}, Transformers {transformers.__version__}, OpenCV {cv2.__version__}")
    print(f"[OK] GPU {properties.name}, VRAM {vram_gib:.2f} GiB")
    if vram_gib < 20:
        print("[WARNING] The unchanged batch_size=64 protocol is intended here for a >=20 GiB GPU such as L4.")
    if shutil.which("java") is None:
        raise RuntimeError("Java is missing; METEOR and SPICE evaluation require it.")
    print("[OK] Java is available for METEOR/SPICE")


def strict_model_smoke_test(config, checkpoint_path, test_json, images_root):
    transformers.logging.set_verbosity_error()
    from Datasets.dataset_utils import getTestTransforms
    from Model import TRCaptionNetpp

    model = TRCaptionNetpp(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("it") != 330000:
        raise ValueError(f"Unexpected public checkpoint iteration: {checkpoint.get('it')}")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    print("[OK] Public checkpoint strict=True load; checkpoint iteration=330000")

    device = torch.device("cuda:0")
    model = model.to(device).eval()
    with open(test_json, "r", encoding="utf-8") as fp:
        sample = json.load(fp)["images"][0]
    sample_path = image_path(images_root, sample["filename"])
    transform = getTestTransforms(model_config=config["model"])
    batch = transform(Image.open(sample_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        caption = model.generate(batch)[0]
    print(f"[OK] End-to-end GPU generation: {caption}")


def main():
    parser = argparse.ArgumentParser(description="Validate the complete TRCaptionNet++ TasvirEt Colab baseline.")
    parser.add_argument("--config", default="configs/tasviret/tasviretpp_large_tasviret.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/TRCaptionNetpp_Large.pth")
    parser.add_argument("--data-dir", default="Data/tasvir-et")
    parser.add_argument("--images-root", default="Data/flickr8k/images")
    parser.add_argument("--skip-model-smoke-test", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    images_root = Path(args.images_root)
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    check_runtime()
    check_config(config)
    check_checkpoint(checkpoint_path)
    check_data(data_dir, images_root)
    if not args.skip_model_smoke_test:
        strict_model_smoke_test(
            config,
            checkpoint_path,
            data_dir / "tasvir_test.json",
            images_root,
        )
    print("PREFLIGHT PASSED: baseline is ready for training.")


if __name__ == "__main__":
    main()
