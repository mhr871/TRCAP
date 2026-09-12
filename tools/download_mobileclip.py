import argparse
import urllib.request
from pathlib import Path

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip"
VALID_MODELS = ("mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b")
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Download an official Apple MobileCLIP checkpoint.")
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--output", default=None,
                        help="Defaults to checkpoints/<model>.pt")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = repo_path(args.output or f"checkpoints/{args.model}.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and output.stat().st_size > 0 and not args.force:
        print(f"Checkpoint already exists: {output} ({output.stat().st_size} bytes)")
        return

    url = f"{BASE_URL}/{args.model}.pt"
    print(f"Downloading {url} to {output}")
    urllib.request.urlretrieve(url, str(output))
    print(f"Done: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
