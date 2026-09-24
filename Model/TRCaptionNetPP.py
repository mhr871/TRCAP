import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, ElectraForCausalLM

from Model.dino import DinoV2
from models.projection_adapters import build_projection_adapter


class TRCaptionNetPP(nn.Module):
    """DINOv2 (frozen) -> Projection Adapter (trainable) -> ELECTRA decoder (trainable).

    Only the projection adapter's architecture changes across the P1-P7
    experiments (see models/projection_adapters); the vision encoder and
    text decoder are held fixed by design, per the Projection Adapter
    ablation study (TRCaptionNet_Projection_Adapter_Deney_Dokumani.docx).

    The decoder is built from transformers.ElectraForCausalLM with
    is_decoder=True and add_cross_attention=True rather than the
    project's own Model/bert/med.py BertLMHeadModel: ELECTRA-base's
    embedding_size equals its hidden_size (768), so its pretrained
    self-attention/FFN weights load directly under HF's own "electra.*"
    parameter names, and add_cross_attention=True adds the extra
    (randomly initialized, trainable) cross-attention sub-layers the
    same way BLIP-style BERT decoders do. Loading the same checkpoint
    into a BERT-shaped module instead would silently fail to match any
    "electra.*" key and leave the whole decoder randomly initialized.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.max_length = config["max_length"]
        self.freeze_encoder = config.get("freeze_encoder", True)
        self._checked_caption_sep = False

        # vision encoder (frozen, per the Projection Adapter ablation design)
        self.vision_encoder = DinoV2(config["dino2"])
        encoder_output_size = self.vision_encoder.get_output_dim()
        if self.freeze_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad_(False)
            self.vision_encoder.eval()

        # text decoder
        electra_id = config["electra"]
        decoder_config = AutoConfig.from_pretrained(electra_id)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        self.language_decoder = ElectraForCausalLM.from_pretrained(electra_id, config=decoder_config)
        self.tokenizer = AutoTokenizer.from_pretrained(electra_id)
        self.train_decoder = bool(config.get("train_decoder", True))
        if not self.train_decoder:
            for p in self.language_decoder.parameters():
                p.requires_grad_(False)
            self.language_decoder.eval()

        # projection adapter (the only part that changes between P1-P7)
        adapter_name = config["projection_adapter"]
        adapter_kwargs = config.get("projection_adapter_kwargs") or {}
        self.proj = build_projection_adapter(adapter_name, input_dim=encoder_output_size,
                                              output_dim=decoder_config.hidden_size, **adapter_kwargs)
        self.train_projection = bool(config.get("train_projection", True))
        if not self.train_projection:
            for p in self.proj.parameters():
                p.requires_grad_(False)
        return

    def encode_image(self, images):
        if self.freeze_encoder:
            with torch.no_grad():
                image_embeds = self.vision_encoder(images).float()
        else:
            image_embeds = self.vision_encoder(images).float()
        return self.proj(image_embeds)

    def forward(self, images, captions, return_acc: bool = False):
        image_embeds = self.encode_image(images)
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

        captions.input_ids[:, 0] = self.tokenizer.cls_token_id
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
        if not return_acc:
            return loss_lm

        # Teacher-forced next-token accuracy, for optional training-time
        # monitoring alongside loss. Must mirror ElectraForCausalLM's own
        # internal shift EXACTLY (logits[:, :-1, :] vs labels[:, 1:]) or
        # this silently measures predictions against the wrong target
        # position. Reuses decoder_output.logits from this same forward
        # pass -- no extra decoder call -- and is computed under no_grad so
        # it cannot affect the loss's gradient graph.
        with torch.no_grad():
            shifted_logits = decoder_output.logits[:, :-1, :]
            shifted_targets = decoder_targets[:, 1:]
            valid = shifted_targets != -100
            preds = shifted_logits.argmax(dim=-1)
            correct = (preds == shifted_targets) & valid
            acc = correct.sum().float() / valid.sum().clamp(min=1).float()
        return loss_lm, acc

    @torch.no_grad()
    def generate(self, images, max_length: int = None, min_length: int = 12, num_beams: int = 3,
                 repetition_penalty: float = 1.1, return_token_ids: bool = False):
        image_embeds = self.encode_image(images)

        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long).to(images.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        input_ids = torch.ones((image_embeds.shape[0], 1), device=images.device, dtype=torch.long)
        input_ids *= self.tokenizer.cls_token_id

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
