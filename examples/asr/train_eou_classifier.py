import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import EOUDetectionModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="token_classification_config")
def main(cfg: DictConfig) -> None:
    trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')

    model = EOUDetectionModel(cfg.model, trainer=trainer)

    # eval
    if cfg.model.train_ds is None and cfg.model.test_ds is not None:
        model.maybe_init_from_pretrained_checkpoint(cfg)
        logging.info("Model restoration success. Starting test...")
        trainer.test(model)
    # train
    else:
        trainer.fit(model)


if __name__ == "__main__":
    main()
