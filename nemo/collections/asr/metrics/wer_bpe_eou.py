from typing import List, Tuple

import re
import editdistance
import torch
from torchmetrics import Metric

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging
from nemo.collections.asr.metrics.wer_bpe import WERBPE


class WERBPEEOU(WERBPE):
    """
    Has the same functionality as the metric WERBPE. However, to focus on measuring the peformance of the model on
    detecting End-of-Utterance, this metric converts all non EOU tokens to the token with ID 0 in the tokenizer for
    WER evaluation.
    """

    def __init__(
            self,
            tokenizer: TokenizerSpec,
            batch_dim_index=0,
            use_cer=False,
            ctc_decode=True,
            log_prediction=True,
            dist_sync_on_step=False,
            eou_token="##\u00a5",
            sub_token_id=0
    ) -> None:
        """
        Utilizes WERBPE __init__.
        In addition, this method looks for the the EOU token's ID in the tokenizer and saves it.

        Args:
          eou_token: String representation of the EOU token
          sub_token_id: Index of token to be used to replace all non EOU tokens during decoding
        """
        super().__init__(tokenizer, batch_dim_index, use_cer, ctc_decode, log_prediction, dist_sync_on_step)

        self.vocabulary = tokenizer.tokenizer.get_vocab()
        if eou_token not in self.vocabulary:
            raise ValueError(f"{eou_token} not found in the tokenizer's vocabulary.")

        self.eou_token = eou_token

        for char in self.vocabulary:
            if self.vocabulary[char] == sub_token_id:
                self.sub_token = char
                break

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Decodes a token list into a string. However, replaces all non EOU tokens with self.sub_token.

        Args:
            tokens: List of int representing the token ids.
        Returns:
            A list of decoded tokens.
        """
        decoded_tokens = super().decode_tokens_to_str(tokens)
        converted_decoded_str = [
            self.sub_token if char != self.eou_token else char for char in decoded_tokens
        ]

        substituted_str = "".join(converted_decoded_str)

        return re.sub(f"{self.sub_token}{2,}", f"{self.sub_token}", substituted_str)
