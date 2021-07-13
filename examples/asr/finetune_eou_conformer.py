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


def read_manifest(path: str) -> List[Dict[str, Any]]:
    """
    Reads the manifest file from <path> and returns a list of dictionaries each containing the metadata of the
    corresponding data sample.

    Argument(s):
        path: string pointing to path of manifest file.
    """
    manifest = []
    with open(path, "r") as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def write_processed_manifest(data: List[Dict[str, Any]], original_path: str) -> str:
    """
    Write the processed manifest data, using <original_path> as the base of the name of the new file.

    Argument(s):
        data: the new and processed list of dictionaries each containing the metadata of an audio file sample and the
            corresponding transcript
        original_path: a string pointing to the original path of the unprocessed manifest data.
    """
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, "w+") as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath


def get_charset(manifest_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    From the manifest data given, return a dictionary mapping each character to the number of occurrences of the
    character.

    Argument(s):
        manifest_data: a list of dictionaries each containing the metadata of an audio file sample and the corresponding
            transcript
    """
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row["text"]
        for character in text:
            charset[character] += 1
    return charset


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


def update_model_config(model: nemo_asr.models.ASRModel, cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
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


def setup_opt_sched(model: nemo_asr.models.ASRModel, cfg: omegaconf.DictConfig) -> None:
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


def setup_spec_augment(model: nemo_asr.models.ASRModel, cfg: omegaconf.DictConfig) -> None:
    """
    Set up the Spectrogram Augmentation using the given arguments.
    """
    with open_dict(model.cfg.spec_augment):
        model.cfg.spec_augment.freq_masks = cfg.model.spec_augment.freq_masks
        model.cfg.spec_augment.time_masks = cfg.model.spec_augment.time_masks


def remove_from_regex(data: Dict[str, Any], regex: str) -> Dict[str, Any]:
    """
    Remove any token matching with <regex>.
    """
    data["text"] = re.sub(regex, "", data["text"])

    return data


def apply_preprocessors(manifest: List[Dict[str, Any]], preprocessors: List[Callable]) -> List[Dict[str, Any]]:
    """
    Return the preprocessed manifest by applying the list of preprocessors to it.

    Argument(s):
        manifest: a list of dictionaries each containing the metadata of an audio file sample and the corresponding
            transcript
        preprocessors: a list of Callables that can be called on the individual dictionaries within <manifest> to
            process them
    """
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest!")
    return manifest


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    """
    Set up and run the fine-tune task for EOU detection using pretrained Conformer-CTC model.
    """
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    train_manifest = read_manifest(cfg.model.train_ds.manifest_filepath)
    train_charset = get_charset(train_manifest)
    train_set = set(train_charset.keys())

    os.system(f"python scripts/tokenizers/process_asr_text_tokenizer.py \\\n"
              f"--manifest={cfg.model.train_ds.manifest_filepath} \\\n"
              f"--vocab_size={len(train_set) + 2} \\\n"
              f"--data_root={cfg.model.tokenizer.dir} \\\n"
              f"--tokenizer=spe \\\n"
              f"--spe_type={cfg.model.tokenizer.type} \\\n"
              f"--spe_character_coverage=1.0 \\\n"
              f"--no_lower_case \\\n"
              f"--log")

    tokenizer_dir = f"{cfg.model.tokenizer.dir}/tokenizer_spe_{cfg.model.tokenizer.type}_v{len(train_set) + 2}/"

    new_validation_paths = omegaconf.ListConfig([])
    for validation_manifest_path in cfg.model.validation_ds.manifest_filepath:
        validation_manifest = read_manifest(validation_manifest_path)
        validation_charset = get_charset(validation_manifest)
        validation_set = set(validation_charset.keys())

        train_validation_common = set.intersection(train_set, validation_set)
        validation_oov = validation_set - train_validation_common

        if validation_oov:
            oov_removal_regex = "[" + "".join(token for token in validation_oov) + "]"
            remove_oov = lambda data: remove_from_regex(data, oov_removal_regex)
            preprocessors = [remove_oov]

            validation_data_processed = apply_preprocessors(validation_manifest, preprocessors)
            new_validation_paths.append(write_processed_manifest(
                validation_data_processed, validation_manifest_path))
        else:
            new_validation_paths.append(validation_manifest_path)

    cfg.model.validation_ds.manifest_filepath = new_validation_paths

    model = setup_model(cfg.model.name, cfg.model.freeze_encoder)
    model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type="bpe")

    updated_config = update_model_config(model, cfg)
    trainer = setup_trainer(model, cfg)

    model.setup_training_data(updated_config.train_ds)
    model.setup_multiple_validation_data(updated_config.validation_ds)

    setup_opt_sched(model, cfg)
    setup_spec_augment(model, cfg)

    exp_manager.exp_manager(trainer, cfg.get("exp_manager", None))
    trainer.fit(model)


if __name__ == "__main__":
    main()
