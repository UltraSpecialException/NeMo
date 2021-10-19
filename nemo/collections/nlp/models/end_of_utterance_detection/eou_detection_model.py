import os
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from typing import Optional, List, Any, Union, Dict

from nemo.collections.nlp.models.token_classification import TokenClassificationModel
from nemo.collections.common.losses import FocalLoss
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_labels_to_labels_id_mapping
from nemo.collections.nlp.data.token_classification.token_classification_dataset import (
    BertTokenClassificationDataset,
    BertTokenClassificationInferDataset,
)
from nemo.collections.nlp.data.token_classification.token_classification_utils import get_label_ids
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.parts.utils_funcs import get_classification_report, plot_confusion_matrix, tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging


def generate_eou_loss_mask(text: List[str], tokenizer: Any) -> List[int]:
    """
    Return the loss mask from the given labels where the silence tokens will be be considered towards the loss whereas
    the word tokens will not.
    """
    mask = [0]   # start token
    for word in text:
        if word.isnumeric():
            mask.append(1)
        else:
            word_tokens = tokenizer.text_to_tokens(word)
            mask.extend([0 for _ in range(len(word_tokens))])

    mask.append(0)    # end token
    return mask


class EOUDetectionModel(TokenClassificationModel):
    """
    This model takes input transcribed text from an ASR model with silence durations to classify each silence token as
    either end-of-utterance or not.
    """

    def setup_loss(self, class_balancing: Optional[str] = None):
        """
        Set up loss depending on the type of class balancing strategy given. This method extends the setup_loss method
        in TokenClassificationModel to support focal loss.
        """
        if class_balancing not in ["weighted_loss", "focal_loss", None]:
            raise ValueError(
                f"Class balancing {class_balancing} is not supported. Choose from: [null, weighted_loss, focal_loss]")

        if class_balancing == "focal_loss":
            loss = FocalLoss(alpha=torch.Tensor(self.class_weights), gamma=self._cfg.dataset.focal_loss_gamma)
            logging.debug(f"Using {class_balancing} class balancing.")

        else:
            loss = super().setup_loss(class_balancing)

        return loss

    def _setup_dataloader_from_config(self, cfg: DictConfig) -> DataLoader:
        """
        Setup dataloader from config
        Args:
            cfg: config for the dataloader
        Return:
            Pytorch Dataloader
        """
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")

        text_file = os.path.join(data_dir, cfg.text_file)
        labels_file = os.path.join(data_dir, cfg.labels_file)

        if not (os.path.exists(text_file) and os.path.exists(labels_file)):
            raise FileNotFoundError(
                f'{text_file} or {labels_file} not found. The data should be split into 2 files: text.txt and \
                labels.txt. Each line of the text.txt file contains text sequences, where words are separated with \
                spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are \
                separated with spaces. Each line of the files should follow the format:  \
                   [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
                   [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
            )

        loss_mask_gen_function = generate_eou_loss_mask if self._cfg.dataset.text_only is None or not self._cfg.dataset.text_only else None
        
        dataset = BertTokenClassificationDataset(
            text_file=text_file,
            label_file=labels_file,
            max_seq_length=dataset_cfg.max_seq_length,
            tokenizer=self.tokenizer,
            num_samples=cfg.num_samples,
            pad_label=dataset_cfg.pad_label,
            label_ids=self._cfg.label_ids,
            ignore_extra_tokens=dataset_cfg.ignore_extra_tokens,
            ignore_start_end=dataset_cfg.ignore_start_end,
            use_cache=dataset_cfg.use_cache,
            loss_mask_gen=loss_mask_gen_function
        )
        return DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=dataset_cfg.num_workers,
            pin_memory=dataset_cfg.pin_memory,
            drop_last=dataset_cfg.drop_last,
        )

    def _setup_infer_dataloader(self, queries: List[str], batch_size: int) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Args:
            queries: text
            batch_size: batch size to use during inference
        Returns:
            A pytorch DataLoader.
        """
        loss_mask_gen_function = generate_eou_loss_mask if self._cfg.dataset.text_only is None or not self._cfg.dataset.text_only else None
        dataset = BertTokenClassificationInferDataset(
            tokenizer=self.tokenizer, queries=queries, max_seq_length=-1,
            loss_mask_gen=loss_mask_gen_function)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=False,
        )
