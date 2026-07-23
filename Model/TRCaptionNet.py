import os

import numpy
from PIL import Image
from torchvision import transforms

from Model import clip
import torch
from torch import nn
from transformers import AutoTokenizer, BertTokenizer
from Model.bert import BertLMHeadModel, BertConfig
from Model.clip.model import Transformer
from Model.dino import DinoV2


class Proj(nn.Module):

    def __init__(self, encoder_output_size, num_head=16):
        super().__init__()
        self.encoder_output_size = encoder_output_size

        self.transformer = Transformer(encoder_output_size, 1, num_head)
        self.linear = nn.Linear(encoder_output_size, 768)
        return

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return self.linear(x)


class TRCaptionNetpp(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # parameters
        self.max_length = config["max_length"]
        self.proj_flag = config["proj"]
        assert type(self.proj_flag) is bool
        self.proj_num_head = config["proj_num_head"]
        self._checked_caption_sep = False

        # vision encoder
        if "clip" in config:
            self.vision_encoder, preprocess = clip.load(config["clip"], jit=False)
            self.vision_encoder.eval()
            self.vision_encoder = self.vision_encoder.visual
            with torch.no_grad():
                dummpy_input_image = preprocess(Image.fromarray(numpy.zeros((512, 512, 3), dtype=numpy.uint8))).to(
                    next(self.parameters()).device)
                encoder_output_size = self.vision_encoder(dummpy_input_image.unsqueeze(0)).shape[-1]
        elif "dino2" in config:
            self.vision_encoder = DinoV2(config["dino2"])
            encoder_output_size = self.vision_encoder.get_output_dim()
        else:
            raise Exception("Image Encoder Init Error!")

        # language decoder
        if not os.path.isfile(config["bert"]):
            self.language_decoder = BertLMHeadModel.from_pretrained(config["bert"],
                                                                    is_decoder=True,
                                                                    add_cross_attention=True)
            self.tokenizer = BertTokenizer.from_pretrained(config["bert"])
        else:
            med_config = BertConfig.from_json_file(config["bert"])
            self.language_decoder = BertLMHeadModel(config=med_config)
            self.tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

        # proj
        if self.proj_flag:
            if self.proj_num_head is None:
                self.proj = nn.Linear(encoder_output_size, 768)
            else:
                self.proj = Proj(encoder_output_size, self.proj_num_head)
        else:
            self.proj = None
        return

    def forward(self, images, captions):
        with torch.no_grad():
            image_embeds = self.vision_encoder(images).float().detach()

        image_embeds = self.proj(image_embeds)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)

        captions = self.tokenizer(captions, padding='longest', truncation=True, max_length=self.max_length,
                                  return_tensors="pt").to(images.device)

        if not self._checked_caption_sep:
            valid_lengths = captions.attention_mask.sum(dim=1)
            row_ids = torch.arange(captions.input_ids.size(0), device=captions.input_ids.device)
            last_token_ids = captions.input_ids[row_ids, valid_lengths - 1]
            if not torch.all(last_token_ids == self.tokenizer.sep_token_id).item():
                raise RuntimeError("Caption target check failed: at least one caption does not end with [SEP].")
            print(f"Caption [SEP] check passed: sep_token_id={self.tokenizer.sep_token_id}")
            self._checked_caption_sep = True

        captions.input_ids[:, 0] = 2
        decoder_targets = captions.input_ids.masked_fill(captions.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, 0] = -100

        decoder_output = self.language_decoder(input_ids=captions.input_ids,
                                               attention_mask=captions.attention_mask,
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts,
                                               labels=decoder_targets,
                                               return_dict=True,
                                               )

        loss_lm = decoder_output.loss
        return loss_lm

    @torch.no_grad()
    def generate(self, images, max_length: int = None, min_length: int = 12, num_beams: int = 3,
                 repetition_penalty: float = 1.1, return_token_ids: bool = False):
        image_embeds = self.vision_encoder(images).float()

        if self.proj is not None:
            image_embeds = self.proj(image_embeds)

        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long).to(images.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        input_ids = torch.ones((image_embeds.shape[0], 1), device=images.device, dtype=torch.long)
        input_ids *= 2

        outputs = self.language_decoder.generate(input_ids=input_ids,
                                                 max_length=self.max_length if max_length is None else max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        if return_token_ids:
            return captions, outputs
        return captions
