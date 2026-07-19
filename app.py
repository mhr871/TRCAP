import glob
import os

import gdown
import gradio as gr
import torch
from torchvision import transforms

from Model import TRCaptionNetpp


model_ckpt = "./checkpoints/TRCaptionNetpp_Large.pth"

os.makedirs("./checkpoints/", exist_ok=True)
url = "https://drive.google.com/uc?id=1tOiRtIpe99gQWnpGfy_W5xgtsHFhvU3F"
gdown.download(url, model_ckpt, quiet=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)

model = TRCaptionNetpp(
    {
        "max_length": 35,
        "dino2": "dinov2_vitl14",
        "bert": "dbmdz/electra-base-turkish-mc4-cased-discriminator",
        "proj": True,
        "proj_num_head": 16,
    }
)
ckpt = torch.load(model_ckpt, map_location=device)
model.load_state_dict(ckpt["model"], strict=True)
model = model.to(device)
model.eval()


def inference(raw_image, min_length, repetition_penalty):
    batch = preprocess(raw_image).unsqueeze(0).to(device)
    caption = model.generate(
        batch,
        min_length=int(min_length),
        repetition_penalty=float(repetition_penalty),
    )[0]
    return caption


# ----- UI -----
img_input = gr.Image(type="pil", interactive=True, label="Input Image")
minlen_slider = gr.Slider(
    minimum=6, maximum=22, value=11, step=1, label="MINIMUM CAPTION LENGTH"
)
rep_slider = gr.Slider(
    minimum=1.0, maximum=3.0, value=2.5, step=0.1, label="REPETITION PENALTY"
)

outputs = gr.Textbox(label="Caption")

title = "TRCaptionNet"
paper_link = ""  # add if available
github_link = "https://github.com/serdaryildiz/TRCaptionNetpp"
description = (
    f"<p style='text-align: center'>"
    f"<a href='{github_link}' target='_blank'>TRCaptionNet++</a>: "
    f"A high-performance encoder–decoder based Turkish image captioning model "
    f"fine-tuned with a large-scale pretrain dataset.</p>"
)
article = (
    f"<p style='text-align: center'>"
    f"<a href='{paper_link}' target='_blank'>Paper</a> | "
    f"<a href='{github_link}' target='_blank'>Github Repo</a></p>"
)
css = ".output-image, .input-image, .image-preview {height: 600px !important}"

# Build examples with full rows (image, min_length, repetition_penalty)
imgs = glob.glob("images/*")
if imgs:
    examples = [[p, 11, 2.0] for p in imgs]
    cache_examples = True
else:
    examples = None
    cache_examples = False  # avoid startup caching when there are no examples

iface = gr.Interface(
    fn=inference,
    inputs=[img_input, minlen_slider, rep_slider],
    outputs=outputs,
    title=title,
    description=description,
    examples=examples,
    cache_examples=cache_examples,
    article=article,
    css=css,
)

if __name__ == "__main__":
    # If you still hit caching issues, you can also set: ssr_mode=False
    iface.launch(server_name="0.0.0.0", server_port=7860)
