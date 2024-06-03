#!/usr/bin/python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import BertForMaskedLM,AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertLayer
from transformers.modeling_outputs import MaskedLMOutput
from arguments import ModelArguments

@dataclass
class MaskedLMOutputWithLogs(MaskedLMOutput):
    logs: Optional[Dict[str, any]] = None

class BertForCotMAE(nn.Module):
    def __init__(
        self,
        bert: BertForMaskedLM,
        model_args: ModelArguments,
    ):
        super(BertForCotMAE,self).__init__()
        self.lm = bert
        #self.model_args = model_args

        self.use_decoder_head = model_args.use_decoder_head
        self.n_head_layers = model_args.n_head_layers
        self.enable_head_mlm = model_args.enable_head_mlm
        self.head_mlm_coef = model_args.head_mlm_coef

        if self.use_decoder_head:
            self.c_head = nn.ModuleList(
                [BertLayer(self.lm.config) for _ in range(self.n_head_layers)]
            )
            self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.model_args = model_args

    def forward(self, **model_input):
        lm_out: MaskedLMOutput = self.lm(
            input_ids = model_input['input_ids'],
            attention_mask = model_input['attention_mask'],
            labels=model_input['labels'],
            output_hidden_states=True,
            return_dict=True
        )
        #import pdb;pdb.set_trace()

        cls_hiddens = lm_out.hidden_states[-1][:, 0]  # B,1,D

        logs = dict()

        # add last layer mlm loss
        loss = lm_out.loss
        logs["encoder_mlm_loss"] = lm_out.loss.item()

        if self.use_decoder_head and self.enable_head_mlm:
            # Get the embedding of decoder inputs
            decoder_embedding_output = self.lm.bert.embeddings(input_ids=model_input['decoder_input_ids'])
            decoder_attention_mask = self.lm.get_extended_attention_mask(
                                        model_input['decoder_attention_mask'],
                                        model_input['decoder_attention_mask'].shape,
                                        model_input['decoder_attention_mask'].device
                                    )  # [batch,L] -> [batch,1,1,L]
            # Concat cls-hiddens of span A & embedding of span B
            hiddens = torch.cat([cls_hiddens.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)
            for layer in self.c_head:
                layer_out = layer(
                    hiddens,
                    decoder_attention_mask,
                )
                hiddens = layer_out[0]
            # add head-layer mlm loss
            head_mlm_loss = self.mlm_loss(hiddens, model_input['decoder_labels']) * self.head_mlm_coef
            logs["head_mlm_loss"] = head_mlm_loss.item()
            loss += head_mlm_loss

        return MaskedLMOutputWithLogs(
            loss=loss,
            logits=lm_out.logits,
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
            logs=logs,
        )

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_args: ModelArguments,
            *args, **kwargs):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        #import pdb;pdb.set_trace()
        from peft import get_peft_config, get_peft_model, LoraConfig, TaskType,AdaLoraConfig
        peft_cls = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=['query' ,'key'])
        #peft_cls = AdaLoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=["query","key","value"])

        hf_model.resize_token_embeddings(30522)
        model_cls = get_peft_model(hf_model,peft_cls)
        print(model_cls.print_trainable_parameters())
        model = cls(model_cls, model_args)  # hf_model
        return model
