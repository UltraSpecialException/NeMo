import os
import glob
import subprocess
import tarfile
import omegaconf
import wget
import copy
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


def setup_model(cfg: omegaconf.DictConfig, trainer: ptl.Trainer) -> nemo_asr.models.EncDecCTCModelBPE:
    """
    Returns a model from the given pretrained model name.

    Argument(s):
        cfg: the loaded YAML file containing the configurations necessary
    """
    pretrained_model = nemo_asr.models.ASRModel.from_pretrained(cfg.model.name, map_location="cpu")

    if "run_two_head" in cfg and cfg.run_two_head:
        model_class = nemo_asr.models.EOUDetectionModel
    else:
        model_class = nemo_asr.models.EncDecCTCModelBPE

    model = model_class(cfg=cfg.model, trainer=trainer)

    try:
        model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=False)
        logging.info("Successfully loaded encoder weights")
    except Exception as e:
        logging.info(f"Could not load encoder checkpoint: {e}")

    try:
        model.encoder.load_state_dict(pretrained_model.decoder.state_dict(), strict=False)
        logging.info("Successfully loaded decoder weights")
    except Exception as e:
        logging.info(f"Could not load decoder checkpoint: {e}")

    if "run_two_head" in cfg and cfg.run_two_head:
        model.eou_decoder.load_state_dict(pretrained_model.decoder.state_dict(), strict=False)

    del pretrained_model

    if cfg.model.freeze_encoder:
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


def update_model_config(model: nemo_asr.models.EncDecCTCModelBPE, cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """
    Update the model's inner configuration using the arguments given.

    Argument(s):
        cfg: the loaded YAML file containing the configurations necessary
    """
    # Setup train, validation configs
    with open_dict(cfg):
        # Train dataset
        model.cfg.train_ds.manifest_filepath = cfg.model.train_ds.manifest_filepath
        model.cfg.train_ds.batch_size = cfg.model.train_ds.batch_size
        model.cfg.train_ds.is_tarred = cfg.model.train_ds.is_tarred
        model.cfg.train_ds.tarred_audio_filepaths = cfg.model.train_ds.tarred_audio_filepaths
        model.cfg.train_ds.shuffle_n = cfg.model.train_ds.shuffle_n
        model.cfg.train_ds.num_workers = cfg.model.train_ds.num_workers

        # Validation dataset
        model.cfg.validation_ds.manifest_filepath = cfg.model.validation_ds.manifest_filepath
        model.cfg.validation_ds.batch_size = cfg.model.validation_ds.batch_size
        model.cfg.validation_ds.num_workers = cfg.model.validation_ds.num_workers

    return model.cfg


def setup_opt_sched(model: nemo_asr.models.EncDecCTCModelBPE, cfg: omegaconf.DictConfig) -> None:
    """
    Set up the optimizer's and scheduler's configuration based on the given arguments.

    Argument(s):
        cfg: the loaded YAML file containing the configurations necessary
    """
    with open_dict(model.cfg.optim):
        model.cfg.optim.lr = cfg.model.optim.lr
        model.cfg.optim.name = cfg.model.optim.name
        model.cfg.optim.betas = cfg.model.optim.betas
        model.cfg.optim.weight_decay = cfg.model.optim.weight_decay
        model.cfg.optim.sched.warmup_steps = cfg.model.optim.sched.warmup_steps
        model.cfg.optim.sched.min_lr = cfg.model.optim.sched.min_lr

    model.setup_optimization(model.cfg.optim)


def setup_exp_manager(exp_dir: str, name: str) -> omegaconf.DictConfig:
    """
    Set up the experiment manager using the given arguments.
    """
    config = exp_manager.ExpManagerConfig(
        exp_dir=exp_dir,
        name=name,
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer",
            mode="min",
            always_save_nemo=True,
            save_best_model=True
        )
    )

    config = OmegaConf.structured(config)

    return config


def setup_trainer(model: nemo_asr.models.EncDecCTCModelBPE, cfg: omegaconf.DictConfig) -> ptl.Trainer:
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


def setup_spec_augment(model: nemo_asr.models.EncDecCTCModelBPE, cfg: omegaconf.DictConfig) -> None:
    """
    Set up the Spectrogram Augmentation using the given arguments.
    """
    with open_dict(model.cfg.spec_augment):
        model.cfg.spec_augment.freq_masks = cfg.model.spec_augment.freq_masks
        model.cfg.spec_augment.time_masks = cfg.model.spec_augment.time_masks

    model.spec_augment = model.from_config_dict(model.cfg.spec_augment)


def remove_from_regex(data: Dict[str, Any], regex: str) -> Dict[str, Any]:
    """
    Remove any token matching with <regex>.
    """
    data["text"] = re.sub(regex, "", data["text"])

    return data


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    """
    Set up and run the fine-tune task for EOU detection using pretrained Conformer-CTC model.
    """
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    trainer = ptl.Trainer(**cfg.trainer)
    model = setup_model(cfg, trainer)

    exp_manager.exp_manager(trainer, cfg.get("exp_manager", None))
    trainer.fit(model)


if __name__ == "__main__":
    main()
