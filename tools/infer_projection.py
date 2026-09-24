"""Load any P1-P7 best checkpoint and caption an image, from a single Colab
cell -- no git/CLI round-trip needed to switch models or images.

Usage (paste into a Colab cell, after cloning the repo and mounting Drive):

    import sys; sys.path.insert(0, '/content/TRCAP_projection_exp')
    %cd /content/TRCAP_projection_exp
    from tools.infer_projection import load_model, caption_image, compare_all

    model, args = load_model("P4_cross_attention")
    print(caption_image(model, args, "/content/my_image.jpg"))

    # or run every trained adapter on the same image at once:
    compare_all("/content/my_image.jpg")

Switching models is just calling load_model() again with a different name;
loaded models are cached in-memory (MODEL_CACHE) so re-picking a model you
already loaded this session is instant.
"""
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image

from Datasets.dataset_utils import getTestTransforms
from Model import TRCaptionNetpp
from Model.projection_adapters import ADAPTER_CODES
from utils import over_write_args

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "projection_exp"
# Must match run_projection_experiments.py's --save-dir for the run you want
# to load from. Override by assigning tools.infer_projection.SAVE_ROOT
# before calling load_model(), or pass save_root= explicitly.
SAVE_ROOT = Path("/content/drive/MyDrive/TRCAP_projection_exp")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ADAPTER_NAMES = [f"{code}_{name}" for name, code in ADAPTER_CODES.items()]  # P1_linear, P2_mlp, ...

MODEL_CACHE = {}


def load_model(adapter_choice: str, save_root: Path = None, use_cache: bool = True):
    """adapter_choice: one of ADAPTER_NAMES, e.g. "P1_linear", "P4_cross_attention"."""
    save_root = Path(save_root) if save_root else SAVE_ROOT
    cache_key = (adapter_choice, str(save_root))
    if use_cache and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    config_path = CONFIG_DIR / f"{adapter_choice}.yaml"
    if not config_path.is_file():
        raise ValueError(f"Unknown adapter_choice {adapter_choice!r}; expected one of {ADAPTER_NAMES}")

    args = Namespace(config=str(config_path))
    over_write_args(args, str(config_path))

    ckpt_path = save_root / adapter_choice / f"{adapter_choice}_best.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path} "
                                f"(did run_projection_experiments.py finish this adapter yet?)")

    model = TRCaptionNetpp(args.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(DEVICE).eval()

    print(f"loaded {adapter_choice} from {ckpt_path} "
         f"(iter {checkpoint.get('it')}, best {checkpoint.get('target_metric')}={checkpoint.get('best_eval_val')})")

    result = (model, args)
    if use_cache:
        MODEL_CACHE[cache_key] = result
    return result


@torch.no_grad()
def caption_image(model, args, image_path: str, num_beams: int = 3, max_length: int = 35) -> str:
    transform = getTestTransforms(model_config=args.model)
    image = Image.open(image_path).convert("RGB")
    image_t = transform(image).unsqueeze(0).to(DEVICE)
    caption = model.generate(image_t, max_length=max_length, num_beams=num_beams)[0]
    return caption


def compare_all(image_path: str, save_root: Path = None, adapters=None):
    """Caption the same image with every trained P1-P7 adapter (whichever
    checkpoints already exist) and print them side by side."""
    adapters = adapters or ADAPTER_NAMES
    results = {}
    for adapter_choice in adapters:
        try:
            model, args = load_model(adapter_choice, save_root=save_root)
            results[adapter_choice] = caption_image(model, args, image_path)
        except FileNotFoundError as exc:
            results[adapter_choice] = f"[skipped: {exc}]"
    print(f"\nimage: {image_path}")
    for adapter_choice, caption in results.items():
        print(f"  {adapter_choice:<20} {caption}")
    return results
