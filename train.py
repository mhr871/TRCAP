import argparse
import os
import random

import numpy
import torch
import yaml
from torch.backends import cudnn

from models.projection_adapters import ADAPTER_CODES
from trainer import Trainer
from utils import TBLog, get_logger, over_write_args


def main(opt):
    # random seed
    assert opt.seed is not None
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    numpy.random.seed(opt.seed)
    cudnn.deterministic = True

    adapter_name = opt.model["projection_adapter"]
    run_name = getattr(opt, "save_name", None) or f"{ADAPTER_CODES[adapter_name]}_{adapter_name}"
    save_path = os.path.join(opt.save_dir, run_name)
    opt.save_name = run_name
    opt.save_path = save_path

    # Only block on an actual checkpoint, not a bare existing directory: a
    # run that crashed before ever saving last.pth leaves behind a directory
    # with just logs, and that should be safe to reuse on a plain restart
    # without needing --overwrite or --resume.
    existing_checkpoint = os.path.join(save_path, 'last.pth')
    if os.path.exists(existing_checkpoint) and not opt.overwrite and not opt.resume_ckpt:
        raise Exception('already existing run: {}'.format(existing_checkpoint))

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w', encoding='utf-8') as fp:
        yaml.safe_dump({k: v for k, v in vars(opt).items() if k != 'save_path'},
                       fp, allow_unicode=True, sort_keys=False)

    # set logger
    tb_logger = TBLog(save_path, 'tensorboard', True)
    logger = get_logger(opt.save_name, save_path, "INFO", filename="train.log")
    logger.warning(f"USE GPU: {opt.gpu} for training")
    logger.info(opt)
    trainer = Trainer(args=opt, tb_logger=tb_logger, logger=logger)
    trainer()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRCaptionNet++ Projection Adapter deneyleri')
    parser.add_argument('--config', type=str, required=True,
                        help='e.g. configs/projection_exp/P1_linear.yaml')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Override the config output root, e.g. a mounted Google Drive directory.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume this run from a full last.pth checkpoint.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow reusing a run directory that already has a checkpoint.')
    args = parser.parse_args()
    cli_save_dir = args.save_dir
    cli_resume = args.resume
    cli_overwrite = args.overwrite
    over_write_args(args, args.config)
    if cli_save_dir:
        args.save_dir = cli_save_dir
    args.overwrite = cli_overwrite or bool(getattr(args, 'overwrite', False))
    args.resume_ckpt = cli_resume
    main(args)
