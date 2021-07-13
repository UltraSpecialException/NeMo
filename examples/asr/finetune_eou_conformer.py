import os
import glob
import subprocess
import tarfile
import omegaconf
from omegaconf import OmegaConf, open_dict
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager
from nemo.core.config import hydra_runner
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import *
from collections import defaultdict
import pytorch_lightning as ptl
from argparse import ArgumentParser, Namespace
import re
from string import punctuation
import json


def setup_model(model_name: str, freeze: bool) -> nemo_asr.models.ASRModel:
    """
    Returns a model from the given pretrained model name.

    Argument(s):
        model_name: string of name of the pretrained model to load
        freeze: boolean indicating whether or not to freeze the encoder
    """
    model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location="cpu")

    if freeze:
        def enable_bn_se(module):
            """
            Function to unfreeze the batch norm and SqueezeExcite modules in the encoder when the rest is frozen
            """
            if isinstance(module, nn.BatchNorm1d):
                module.train()
                for param in module.parameters():
                    param.requires_grad_(True)

            if "SqueezeExcite" in type(module).__name__:
                module.train()
                for param in module.parameters():
                    param.requires_grad_(True)

        model.encoder.freeze()
        model.encoder.apply(enable_bn_se)
    else:
        model.encoder.unfreeze()

    return model


def setup_trainer(model: nemo_asr.models.ASRModel, cfg: omegaconf.DictConfig) -> ptl.Trainer:
    """
    Set up the trainer using PyTorch Lightning.

    Argument(s):
        cfg: the loaded YAML file containing the configurations necessary
    """
    if not torch.cuda.is_available():
        logging.info("CUDA unavailable, setting the number of GPUs to be 0.")
        cfg.trainer.gpus = 0

    trainer = ptl.Trainer(**cfg.trainer)

    # Setup model with the trainer
    model.set_trainer(trainer)

    # Finally, update the model's internal config
    model.cfg = model._cfg

    return trainer


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    """
    Set up and run the fine-tune task for EOU detection using pretrained Conformer-CTC model.
    """
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    model = setup_model(cfg.model.name, cfg.model.freeze_encoder)
    model.change_vocabulary(new_tokenizer_dir=cfg.model.tokenizer.dir, new_tokenizer_type=cfg.model.tokenizer.type)

    pretrained_decoder = model.decoder.state_dict()
    if model.decoder.decoder_layers[0].weight.shape == pretrained_decoder["decoder_layers.0.weight"].shape:
        model.decoder.load_state_dict(pretrained_decoder)
        logging.info("Loaded decoder's weights")
    else:
        logging.info("Weights' shape mismatch. Cannot load decoder's weights.")

    trainer = setup_trainer(model, cfg)

    model.setup_training_data(cfg.model.train_ds)
    model.setup_multiple_validation_data(cfg.model.validation_ds)
    model.setup_optimization(cfg.model.optim)
    model.spec_augment = model.from_config_dict(cfg.model.spec_augment)

    exp_manager.exp_manager(trainer, cfg.get("exp_manager", None))
    trainer.fit(model)


if __name__ == "__main__":
    main()
