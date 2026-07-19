import argparse
import os
import random

import numpy
import torch
from torch.backends import cudnn

from trainer import Trainer
from utils import TBLog, get_logger, over_write_args


def main(opt):
    # random seed
    assert opt.seed is not None
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    numpy.random.seed(opt.seed)
    cudnn.deterministic = True

    save_path = os.path.join(opt.save_dir, opt.save_name)
    opt.save_path = save_path
    if os.path.exists(save_path) and not opt.overwrite and not opt.resume_ckpt:
        raise Exception('already existing model: {}'.format(save_path))

    # set logger
    tb_logger = TBLog(save_path, 'tensorboard', True)
    logger_level = "INFO"
    logger = get_logger(opt.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {opt.gpu} for training")
    logger.info(opt)
    trainer = Trainer(args=opt, tb_logger=tb_logger, logger=logger)
    trainer()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TR-CLIP-Captioning!')
    parser.add_argument('--config', type=str, default='./configs/tasviret/tasviretpp_large_tasviret.yaml')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Override the config output root, e.g. a mounted Google Drive directory.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume a fine-tuning run from a full model_last.pth checkpoint.')
    args = parser.parse_args()
    cli_save_dir = args.save_dir
    cli_resume = args.resume
    over_write_args(args, args.config)
    if cli_save_dir:
        args.save_dir = cli_save_dir
    args.resume_ckpt = cli_resume
    main(args)
