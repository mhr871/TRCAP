import json
import os
from logging import Logger

import torch
import tqdm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

from Datasets.dataset_utils import getTrainDataset, getTestDataset
from Model import TRCaptionNetpp
from eval import evaluate_on_coco_caption, predict
from utils import TBLog
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


class Trainer:
    def __init__(self, args, tb_logger: TBLog = None, logger: Logger = None):

        # initialize parameters
        self.args = args
        self.experiment_root = args.save_path
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.device = torch.device(f"cuda:{args.gpu}")
        # self.device = 'cpu'
        self.lr = float(args.lr)
        self.lr_proj = float(args.lr_proj)
        self.betas = args.betas
        self.weight_decay = args.weight_decay
        self.max_iter = args.max_iter
        self.warm_up_iter = args.warm_up_iter
        self.target_metric = args.target_metric
        self.init_model_ckpt = getattr(args, "init_model_ckpt", None)
        self.strict_init = getattr(args, "strict_init", True)
        self.resume_ckpt = getattr(args, "resume_ckpt", None)
        self.last_grad_norm = None
        self.it = 0
        self.best_eval_val = -1
        self.best_it = -1

        # dataset parameters
        self.train_dataset_name = args.train_dataset_name
        self.test_dataset_name = args.test_dataset_name
        self.train_dataset_root = args.train_dataset_root
        self.test_dataset_root = args.test_dataset_root
        self.train_json_path = args.train_json_path
        self.val_json_path = args.val_json_path

        # set tensorboard logger
        self.tb_logger = tb_logger

        # set logger function
        self.logger_fn = logger.info if logger is not None else print
        self.logger_fn(f"USE: {self.device} for training")
        return

    def __call__(self):

        # set dataloaders
        self.train_loader, self.test_loader = self.getDataloaders()

        # initialize model
        self.model = TRCaptionNetpp(self.args.model)
        if self.init_model_ckpt and not self.resume_ckpt:
            checkpoint = torch.load(self.init_model_ckpt, map_location="cpu")
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=self.strict_init)
            self.logger_fn(f"initialized model weights from {self.init_model_ckpt}")
            del state_dict, checkpoint
        self.model = self.model.to(self.device)
        self.log_special_token_ids()

        # initialize optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'name': 'decoder_decay',
             'params': [p for n, p in self.model.language_decoder.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay, "lr": self.lr},
            {'name': 'decoder_no_decay',
             'params': [p for n, p in self.model.language_decoder.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.lr},
            {'name': 'proj_decay',
             'params': [p for n, p in self.model.proj.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay, "lr": self.lr_proj},
            {'name': 'proj_no_decay',
             'params': [p for n, p in self.model.proj.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.lr_proj},

        ]
        # self.model = torch.compile(self.model)

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=self.betas)
        self.validate_optimizer_param_groups()

        # initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.warm_up_iter, self.max_iter)
        self.logger_fn(f"scheduler: linear warmup for {self.warm_up_iter} iterations, then linear decay to 0")

        if self.resume_ckpt:
            self.load_training_checkpoint(self.resume_ckpt)

        self.logger_fn("Train is starting...")
        self.train()
        return

    def train(self):
        # train
        self.model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()

        remaining_iters = max(0, self.max_iter - self.it)
        tbar = tqdm.tqdm(total=remaining_iters, colour='BLUE')
        for image, caption, ids in self.train_loader:
            if self.it >= self.max_iter:
                break
            tbar.update(1)
            self.it += 1

            end_batch.record()
            start_run.record()

            image = image.to(self.device)
            loss = self.model(image, caption)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.last_grad_norm = float(grad_norm.detach().cpu().item())
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/loss'] = loss.detach().cpu().item()
            current_lrs = self.get_current_lrs()
            tb_dict['lr'] = current_lrs['decoder_lr']
            tb_dict['lr/decoder'] = current_lrs['decoder_lr']
            tb_dict['lr/proj'] = current_lrs['proj_lr']
            tb_dict['train/grad_norm'] = self.last_grad_norm
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.args.num_eval_iter == 0:
                eval_dict = self.eval(self.it)
                tb_dict.update(eval_dict)

                if eval_dict[self.target_metric] > self.best_eval_val:
                    self.best_eval_val = eval_dict[self.target_metric]
                    self.best_it = self.it
                    self.save_model('model_best.pth')

                # Keep a resumable checkpoint at every validation boundary.
                self.save_model('model_last.pth')

                self.logger_fn(f"\n {self.it} iteration, {eval_dict},"
                               f" \n BEST {self.target_metric}: {self.best_eval_val}, at {self.best_it} iters")
                self.logger_fn(f" {self.it} iteration, {self.target_metric}:"
                               f" {eval_dict[self.target_metric]}\n")

            if self.tb_logger is not None:
                self.tb_logger.update(tb_dict, self.it)
            del tb_dict
            start_batch.record()

        self.save_model('model_last.pth')
        return

    def log_special_token_ids(self):
        tokenizer = self.model.tokenizer
        decoder_config = self.model.language_decoder.config
        self.logger_fn(
            "tokenizer ids: "
            f"cls_token_id={tokenizer.cls_token_id}, "
            f"sep_token_id={tokenizer.sep_token_id}, "
            f"pad_token_id={tokenizer.pad_token_id}"
        )
        self.logger_fn(
            "decoder config ids: "
            f"bos_token_id={decoder_config.bos_token_id}, "
            f"eos_token_id={decoder_config.eos_token_id}, "
            f"pad_token_id={decoder_config.pad_token_id}"
        )
        return

    def validate_optimizer_param_groups(self):
        decoder_lrs = sorted({group['lr'] for group in self.optimizer.param_groups
                              if group.get('name', '').startswith('decoder')})
        proj_lrs = sorted({group['lr'] for group in self.optimizer.param_groups
                           if group.get('name', '').startswith('proj')})

        if decoder_lrs != [self.lr]:
            raise ValueError(f"Decoder LR mismatch: expected {self.lr}, got {decoder_lrs}")
        if proj_lrs != [self.lr_proj]:
            raise ValueError(f"Projection LR mismatch: expected {self.lr_proj}, got {proj_lrs}")

        self.logger_fn(f"optimizer LR groups verified: decoder_lr={self.lr}, proj_lr={self.lr_proj}")
        return

    def get_current_lrs(self):
        decoder_lrs = [group['lr'] for group in self.optimizer.param_groups
                       if group.get('name', '').startswith('decoder')]
        proj_lrs = [group['lr'] for group in self.optimizer.param_groups
                    if group.get('name', '').startswith('proj')]
        return {
            'decoder_lr': decoder_lrs[0] if decoder_lrs else self.optimizer.param_groups[0]['lr'],
            'proj_lr': proj_lrs[0] if proj_lrs else self.optimizer.param_groups[-1]['lr'],
        }

    def getDataloaders(self):

        # load train dataset
        if type(self.train_dataset_name) == str:
            assert type(self.train_dataset_root) == str
            assert type(self.train_json_path) == str

            train_dataset = getTrainDataset(self.train_dataset_name, self.train_dataset_root, self.train_json_path,
                                            model_config=self.args.model)

        elif type(self.train_dataset_name) == list:
            assert type(self.train_dataset_root) == list
            assert type(self.train_json_path) == list
            train_datasets = []

            for i in range(len(self.train_dataset_name)):
                train_dataset = getTrainDataset(self.train_dataset_name[i],
                                                self.train_dataset_root[i],
                                                self.train_json_path[i],
                                                model_config=self.args.model)
                train_datasets.append(train_dataset)

            train_dataset = ConcatDataset(train_datasets)
        else:
            raise Exception("What do u want to do!! ")

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  sampler=RandomSampler(data_source=train_dataset,
                                                        replacement=True,
                                                        num_samples=self.args.max_iter * self.args.batch_size),
                                  pin_memory=True, drop_last=True)

        # load test dataset
        test_dataset = getTestDataset(self.test_dataset_name, self.test_dataset_root,
                                      self.val_json_path, model_config=self.args.model)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 shuffle=False)
        return train_loader, test_loader

    def eval(self, iter=-1):
        self.model.eval()
        self.logger_fn("Start evaluating")
        val_result, eval_diagnostics = predict(self.model, self.test_loader, self.device, return_diagnostics=True)
        self.save_result(val_result, f"prediction_{iter}.json")
        result = evaluate_on_coco_caption(os.path.join(self.experiment_root, f"prediction_{iter}.json"),
                                          self.val_json_path,
                                          os.path.join(self.experiment_root, f"result_{iter}.json"))
        result['avg_caption_len'] = eval_diagnostics['avg_caption_len']
        result['eos_rate'] = eval_diagnostics['eos_rate']
        self.save_result(result, f"result_{iter}.json")
        current_lrs = self.get_current_lrs()
        self.logger_fn(
            f"eval diagnostics at {iter}: "
            f"Bleu_4={result.get('Bleu_4')}, "
            f"CIDEr={result.get('CIDEr')}, "
            f"avg_caption_len={result['avg_caption_len']:.3f}, "
            f"eos_rate={result['eos_rate']:.3f}, "
            f"decoder_lr={current_lrs['decoder_lr']:.8f}, "
            f"proj_lr={current_lrs['proj_lr']:.8f}, "
            f"grad_norm={self.last_grad_norm}"
        )
        for index, sample_caption in enumerate(eval_diagnostics['sample_captions'], start=1):
            self.logger_fn(f"sample_caption_{index}: {sample_caption}")
        self.logger_fn(result)
        self.model.train()
        return result

    def save_model(self, model_name: str):
        save_filename = os.path.join(self.experiment_root, model_name)
        temp_filename = save_filename + '.tmp'
        self.model.eval()
        save_obj = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'it': self.it,
            'best_eval_val': self.best_eval_val,
            'best_it': self.best_it,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all(),
        }
        torch.save(save_obj, temp_filename)
        os.replace(temp_filename, save_filename)
        self.model.train()
        self.logger_fn(f"model saved: {save_filename}\n")
        return

    def load_training_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.best_eval_val = checkpoint.get('best_eval_val', -1)
        self.best_it = checkpoint.get('best_it', -1)
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state_all' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        del checkpoint
        self.logger_fn(f'training resumed from {load_path} at iteration {self.it}')
        return

    def save_result(self, result, filename):
        result_file = os.path.join(self.experiment_root, '%s' % filename)
        json.dump(result, open(result_file, 'w'))
        return
