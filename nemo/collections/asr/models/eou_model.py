# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer_bpe_eou import WERBPEEOU
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging


class EOUDetectionModel(EncDecCTCModelBPE):
    """
    Encoder decoder based model for End-of-Utterance detection.

    This model is comprised of a regular EncDecCTCModelBPE module (encoder and decoder) responsible for the normal ASR
    task as well as an extra decoder responsible for the task of EOU detection. When training this model, the encoder
    and decoder for regular ASR will be frozen. The ASR model's decoder's logits will be piped into the extra decoder
    for EOU detection.
    """

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self._freeze_asr_model()

        with open_dict(self._cfg):
            if "eou_decoder" not in self._cfg:
                logging.info("No EOU decoder configuration was detected, copying the regular decoder's configuration.")
                self._cfg.eou_decoder = copy.deepcopy(self._cfg.decoder)
                self._cfg.eou_decoder.feat_in = self.decoder.num_classes_with_blank
                self._cfg.eou_decoder.num_classes = self.decoder.num_classes_with_blank - 1

        self.eou_decoder = EncDecCTCModelBPE.from_config_dict(self._cfg.eou_decoder)
        self.loss = CTCLoss(
            num_classes=self.eou_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch")
        )

        wer_eou_args = {}
        if "eou_token" in cfg.tokenizer:
            wer_eou_args["eou_token"] = cfg.tokenizer.eou_token
        else:
            logging.info("Using default EOU token: \u00a5")

        if "sub_token_id" in cfg.tokenizer:
            wer_eou_args["sub_token_id"] = cfg.tokenizer.sub_token_id
        else:
            logging.info("Using token with ID 0 in place of non-EOU tokens.")

        self._wer = WERBPE(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
            **wer_eou_args
        )

    def _freeze_asr_model(self) -> None:
        """
        Called in __init__ to freeze the encoder and decoder of the ASR part of the model.
        """
        self.encoder.freeze()
        self.decoder.freeze()

    @typecheck
    def forward(
            self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model. This extends the forward pass from EncDecCTCModel (EncDecCTCModelBPE's super class).
        After doing the regular forward pass, the logits from the decoder will be passed to the EOU decoder for
        EOU detection.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if not (has_input_signal ^ has_processed_signal):
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        decoder_output = self.decoder.decoder_layers(encoder_output=encoded)
        log_probs = self.eou_decoder(decoder_output)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions
