#!/usr/bin/python
# -*- encoding: utf-8 -*-

import logging
import os
import torch

from typing import Dict,Optional
from collections import defaultdict
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)

class TrainerWithLogs(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inject Customised logging behavior
        self.customized_logging_list = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        A neat compute_loss that supports customized logging

        """
        # 继承transformers.trainer 的compute_loss，重写损失函数
        outputs = model(**inputs)
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Inject Customised logging behavior
        try:
            logs: dict = outputs.logs
        except:
            logs = None
        if logs is not None:
            for k, v in logs.items():
                # Set maxlen of list to avoid memory leak, useful when
                # customized_logging_list has not been cleaned correctly
                if len(self.customized_logging_list[k]) < 5000:
                    self.customized_logging_list[k].append(v)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        # Inject Customised logging behavior
        for k, v in self.customized_logging_list.items():
            if len(v) > 0:
                logs[k] = round(sum(v) / len(v), 4)
        self.customized_logging_list.clear()

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        if not hasattr(self.model, 'save_pretrained'):
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            self.model.save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
