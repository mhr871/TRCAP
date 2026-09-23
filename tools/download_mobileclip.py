import argparse
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip"
V1_MODELS = ("mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b")
# Official Apple MobileCLIP2 releases on the Hugging Face Hub (public, not gated).
V2_MODELS = {
    "mobileclip2_s0": ("apple/MobileCLIP2-S0", "mobileclip2_s0.pt"),
    "mobileclip2_s2": ("apple/MobileCLIP2-S2", "mobileclip2_s2.pt"),
}
VALID_MODELS = V1_MODELS + tuple(V2_MODELS)
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def download(model, output, force):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"Checkpoint already exists: {output} ({output.stat().st_size} bytes)")
        return

    if model in V2_MODELS:
        from huggingface_hub import hf_hub_download

        repo_id, filename = V2_MODELS[model]
        print(f"Downloading hf://{repo_id}/{filename} to {output}")
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copyfile(cached, output)
    else:
        url = f"{BASE_URL}/{model}.pt"
        print(f"Downloading {url} to {output}")
        urllib.request.urlretrieve(url, str(output))
    print(f"Done: {output} ({output.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Download official Apple MobileCLIP / MobileCLIP2 checkpoints.")
    parser.add_argument("--model", required=True, nargs="+", choices=VALID_MODELS)
    parser.add_argument("--output", default=None,
                        help="Only valid with a single --model. Defaults to checkpoints/<model>.pt")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output and len(args.model) > 1:
        parser.error("--output can only be used with a single --model")

    for model in args.model:
        download(model, repo_path(args.output or f"checkpoints/{model}.pt"), args.force)


if __name__ == "__main__":
    main()
