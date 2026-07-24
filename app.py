import argparse
import glob
from pathlib import Path

import gdown
import gradio as gr
import torch
import yaml

from Datasets.dataset_utils import getTestTransforms
from Model import TRCaptionNetpp


PUBLIC_CHECKPOINT_URL = "https://drive.google.com/uc?id=1tOiRtIpe99gQWnpGfy_W5xgtsHFhvU3F"
PUBLIC_CHECKPOINT = Path("checkpoints/TRCaptionNetpp_Large.pth")
FINETUNED_COLAB_WEIGHTS = Path(
    "/content/drive/MyDrive/TRCAP_runs/tasviretpp_large_tasviret_50k_lr1e5/model_best.pth"
)
DEFAULT_CONFIG = Path("configs/tasviret/tasviretpp_large_tasviret.yaml")


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() or str(path).startswith("/") else Path(__file__).resolve().parent / path


def default_weights_path():
    if FINETUNED_COLAB_WEIGHTS.exists():
        return FINETUNED_COLAB_WEIGHTS
    return PUBLIC_CHECKPOINT


def ensure_weights(weights_path):
    if weights_path.exists():
        return
    if weights_path.as_posix().endswith(PUBLIC_CHECKPOINT.as_posix()):
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(PUBLIC_CHECKPOINT_URL, str(weights_path), quiet=False)
        return
    raise FileNotFoundError(f"Model weights not found: {weights_path}")


def load_model(config_path, weights_path, device):
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    preprocess = getTestTransforms(model_config=config["model"])
    model = TRCaptionNetpp(config["model"])
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, preprocess


def build_interface(model, preprocess, device):
    def inference(raw_image, min_length, repetition_penalty):
        if raw_image is None:
            return ""
        image = raw_image.convert("RGB")
        batch = preprocess(image).unsqueeze(0).to(device)
        caption = model.generate(
            batch,
            min_length=int(min_length),
            repetition_penalty=float(repetition_penalty),
        )[0]
        return caption

    img_input = gr.Image(type="pil", interactive=True, label="Input Image")
    minlen_slider = gr.Slider(
        minimum=6, maximum=22, value=11, step=1, label="Minimum caption length"
    )
    rep_slider = gr.Slider(
        minimum=1.0, maximum=3.0, value=2.5, step=0.1, label="Repetition penalty"
    )
    outputs = gr.Textbox(label="Caption")

    imgs = glob.glob("images/*")
    examples = [[p, 11, 2.0] for p in imgs] if imgs else None

    return gr.Interface(
        fn=inference,
        inputs=[img_input, minlen_slider, rep_slider],
        outputs=outputs,
        title="TRCaptionNet++",
        description="Turkish image captioning demo.",
        examples=examples,
        cache_examples=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a TRCaptionNet++ Gradio demo.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link, useful on Colab.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = repo_path(args.config)
    weights_path = repo_path(args.weights) if args.weights else repo_path(default_weights_path())
    ensure_weights(weights_path)

    device = torch.device(args.device)
    model, preprocess = load_model(config_path, weights_path, device)
    iface = build_interface(model, preprocess, device)
    iface.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
