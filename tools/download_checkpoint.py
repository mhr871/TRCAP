import argparse
from pathlib import Path

import gdown


TRCAPTIONNETPP_LARGE_URL = "https://drive.google.com/uc?id=1tOiRtIpe99gQWnpGfy_W5xgtsHFhvU3F"
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Download the public TRCaptionNet++ Large checkpoint.")
    parser.add_argument("--output", default="checkpoints/TRCaptionNetpp_Large.pth")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and output.stat().st_size > 0 and not args.force:
        print(f"Checkpoint already exists: {output} ({output.stat().st_size} bytes)")
        return

    print(f"Downloading TRCaptionNet++ Large checkpoint to {output}")
    gdown.download(TRCAPTIONNETPP_LARGE_URL, str(output), quiet=False)
    print(f"Done: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
