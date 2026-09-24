"""Run the P1-P7 projection adapter experiments sequentially in one process
/ Colab session.

Every experiment is independent: it builds a fresh TRCaptionNetPP model
(frozen pretrained DINOv2 + the adapter under test, freshly initialized +
pretrained ELECTRA decoder), a fresh optimizer/scheduler and iteration
counter, trains it, scores it on the TasvirEt test split, and writes its
results under runs/<Pn>_<adapter>/ before the next experiment starts.
Nothing (weights, optimizer, scheduler, iteration state) is carried from
one adapter to another -- the only thing that changes between runs is
`model.projection_adapter` in each config.

Usage (from the repo root):
  python run_projection_experiments.py
  python run_projection_experiments.py --save-dir /content/drive/MyDrive/TRCAP_projection_exp
  python run_projection_experiments.py --adapters linear mlp residual
"""
import argparse
import gc
import os
from argparse import Namespace
from pathlib import Path

import numpy
import random
import torch
import yaml
from torch.backends import cudnn

from models.projection_adapters import ADAPTERS, ADAPTER_CODES

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "projection_exp"


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one(config_path: Path, save_dir: str, overwrite: bool):
    from trainer import Trainer
    from utils import TBLog, get_logger, over_write_args

    args = Namespace(config=str(config_path))
    over_write_args(args, str(config_path))
    if save_dir:
        args.save_dir = save_dir
    args.resume_ckpt = None
    args.overwrite = overwrite or bool(getattr(args, 'overwrite', False))

    adapter_name = args.model["projection_adapter"]
    run_name = getattr(args, "save_name", None) or f"{ADAPTER_CODES[adapter_name]}_{adapter_name}"
    args.save_name = run_name
    args.save_path = str(repo_path(args.save_dir) / run_name)

    existing_checkpoint = os.path.join(args.save_path, 'last.pth')
    if os.path.exists(existing_checkpoint) and not args.overwrite:
        print(f"[{run_name}] skipping: {existing_checkpoint} already exists (pass --overwrite to re-run)")
        return run_name

    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'config.yaml'), 'w', encoding='utf-8') as fp:
        yaml.safe_dump({k: v for k, v in vars(args).items() if k != 'save_path'},
                       fp, allow_unicode=True, sort_keys=False)

    tb_logger = TBLog(args.save_path, 'tensorboard', True)
    logger = get_logger(run_name, args.save_path, "INFO", filename="train.log")
    logger.info(f"===== {run_name} =====")
    logger.info(args)
    trainer = Trainer(args=args, tb_logger=tb_logger, logger=logger)
    trainer()

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    if tb_logger.writer is not None:
        tb_logger.writer.close()
    cleanup_gpu()
    return run_name


def main():
    parser = argparse.ArgumentParser(description="Run the P1-P7 projection adapter experiments.")
    parser.add_argument("--configs-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--save-dir", default=None, help="Override each config's save_dir, e.g. a Drive path.")
    parser.add_argument("--adapters", nargs="+", default=None,
                        help="Subset of adapter names to run, e.g. linear mlp. Default: all P1-P7.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run an adapter whose run directory already has a checkpoint.")
    args = parser.parse_args()

    configs_dir = repo_path(args.configs_dir)
    selected_adapters = args.adapters or list(ADAPTERS)
    unknown = [a for a in selected_adapters if a not in ADAPTERS]
    if unknown:
        parser.error(f"unknown adapters {unknown}; available: {list(ADAPTERS)}")

    run_names = []
    for adapter_name in selected_adapters:
        code = ADAPTER_CODES[adapter_name]
        config_path = configs_dir / f"{code}_{adapter_name}.yaml"
        if not config_path.is_file():
            raise SystemExit(f"missing config for {code}_{adapter_name}: expected {config_path}")
        print(f"\n===== {code}_{adapter_name} =====")
        run_names.append(run_one(config_path, args.save_dir, args.overwrite))

    print("\n===== projection adapter experiments finished =====")
    for name in run_names:
        print(name)


if __name__ == "__main__":
    main()
