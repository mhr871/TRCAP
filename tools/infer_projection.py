"""Load any P1-P7 checkpoint and caption an image, from a single Colab
cell -- no git/CLI round-trip needed to switch models or images.

Usage (paste into a Colab cell, after cloning the repo and mounting Drive):

    %cd /content/TRCAP_projection_exp_all
    from tools.infer_projection import test

    # one call: pick the checkpoint by name, pick an image from YOUR
    # computer, caption it, and show image+caption together
    test("P4_cross_attention")

Just type a different name to switch models (loaded models are cached in
MODEL_CACHE, so re-picking one you already loaded this session is
instant), and call test(...) again to pick a new image each time.

By default test() loads "<name>_best.pth". To load a specific checkpoint
instead (e.g. the final one), append ":<tag>" to the name, where <tag> is
whatever comes after "<name>_" in the checkpoint filename on Drive, e.g.:

    test("P4_cross_attention:best")        # <name>_best.pth (default)
    test("P4_cross_attention:iter_16000")  # <name>_iter_16000.pth

Lower-level building blocks (load_model / caption_image / compare_all /
upload_image) are still available individually if you want more control
than the single test() call gives you.
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
# before calling load_model()/test(), or pass save_root= explicitly.
SAVE_ROOT = Path("/content/drive/MyDrive/TRCAP_projection_exp_all")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ADAPTER_NAMES = [f"{code}_{name}" for name, code in ADAPTER_CODES.items()]  # P1_linear, P2_mlp, ...

MODEL_CACHE = {}


def upload_image(dest_dir: str = "/content/uploaded_images") -> str:
    """Colab-only: opens your browser's local file picker, uploads the
    chosen image from your own computer to this runtime, and returns its
    path here. Use this instead of hardcoding a path that only exists on
    the Colab VM or on Drive."""
    from google.colab import files

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("no file was uploaded")
    filename = next(iter(uploaded))
    dest_path = str(Path(dest_dir) / filename)
    Path(dest_path).write_bytes(uploaded[filename])
    print(f"uploaded: {dest_path}")
    return dest_path


def load_model(adapter_choice: str, save_root: Path = None, ckpt_tag: str = "best", use_cache: bool = True):
    """adapter_choice: one of ADAPTER_NAMES, e.g. "P1_linear", "P4_cross_attention".
    ckpt_tag: which checkpoint file to load -- "best" (default, matches
        "<adapter_choice>_best.pth") or e.g. "iter_16000" (matches
        "<adapter_choice>_iter_16000.pth"), i.e. whatever
        run_projection_experiments.py / trainer.py actually wrote under
        <save_root>/<adapter_choice>/.
    """
    save_root = Path(save_root) if save_root else SAVE_ROOT
    cache_key = (adapter_choice, str(save_root), ckpt_tag)
    if use_cache and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    config_path = CONFIG_DIR / f"{adapter_choice}.yaml"
    if not config_path.is_file():
        raise ValueError(f"Unknown adapter_choice {adapter_choice!r}; expected one of {ADAPTER_NAMES}")

    args = Namespace(config=str(config_path))
    over_write_args(args, str(config_path))

    ckpt_path = save_root / adapter_choice / f"{adapter_choice}_{ckpt_tag}.pth"
    if not ckpt_path.is_file():
        available = sorted(p.name for p in (save_root / adapter_choice).glob("*.pth")) \
            if (save_root / adapter_choice).is_dir() else []
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}\navailable in that folder: {available}")

    model = TRCaptionNetpp(args.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(DEVICE).eval()

    print(f"loaded {adapter_choice} ({ckpt_tag}) from {ckpt_path} "
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


def test(checkpoint_name: str, save_root: Path = None, image_path: str = None, num_beams: int = 3):
    """One-call manual test: load the named checkpoint (cached across
    calls), get an image (uploads from your computer if image_path isn't
    given), caption it, and display the image with the caption as its
    title.

    checkpoint_name: an adapter run name, e.g. "P4_cross_attention"
        (loads "<name>_best.pth"), or "<name>:<tag>" to load a different
        checkpoint file, e.g. "P4_cross_attention:iter_16000".
    image_path: pass an existing local/Drive path to reuse the same image
        across multiple test() calls without re-uploading; omitted ->
        opens the upload dialog every time.
    """
    from PIL import Image as _Image
    import matplotlib.pyplot as plt

    adapter_choice, _, tag = checkpoint_name.partition(":")
    ckpt_tag = tag or "best"

    model, args = load_model(adapter_choice, save_root=save_root, ckpt_tag=ckpt_tag)
    image_path = image_path or upload_image()
    caption = caption_image(model, args, image_path, num_beams=num_beams)

    plt.imshow(_Image.open(image_path))
    plt.axis("off")
    plt.title(f"[{checkpoint_name}] {caption}", fontsize=11, wrap=True)
    plt.show()
    return caption
