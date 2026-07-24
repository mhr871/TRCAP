import argparse
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Datasets.dataset_utils import getTestTransforms
from Model import TRCaptionNetpp


FINETUNED_COLAB_WEIGHTS = Path(
    "/content/drive/MyDrive/TRCAP_runs/tasviretpp_large_tasviret_50k_lr1e5/model_best.pth"
)
PUBLIC_CHECKPOINT = REPO_ROOT / "checkpoints" / "TRCaptionNetpp_Large.pth"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tasviret" / "tasviretpp_large_tasviret.yaml"


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() or str(path).startswith("/") else REPO_ROOT / path


def default_weights_path():
    if FINETUNED_COLAB_WEIGHTS.exists():
        return FINETUNED_COLAB_WEIGHTS
    return PUBLIC_CHECKPOINT


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_model(config, weights_path, device):
    model = TRCaptionNetpp(config["model"])
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Turkish caption for one image.")
    parser.add_argument("--image", required=True, help="Image file path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-length", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = repo_path(args.image)
    config_path = repo_path(args.config)
    weights_path = repo_path(args.weights) if args.weights else repo_path(default_weights_path())

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    config = load_config(config_path)
    device = torch.device(args.device)
    transform = getTestTransforms(model_config=config["model"])
    model = load_model(config, weights_path, device)

    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)
    caption = model.generate(
        batch,
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )[0]
    print(caption)


if __name__ == "__main__":
    main()
