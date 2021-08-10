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

from typing import List, Set, Tuple

import editdistance
import torch
from torchmetrics import Metric

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging


class EOUMetrics(Metric):
    """
    Computes EOU metrics, including accuracy, precision, recall, F1 & latency.

    Let K be the "acceptable" distance between the predicted and expected EOU token's position
    Accuracy:
    Let E(s) = EOU token position in the reference
        P(s) = EOU token position in the hypothesis

    Then, for sentence s, if |E(s) - P(s)| <= K, then the EOU detection is deemed acceptable. Otherwise, it is not.
    """
    def __init__(
            self,
            tolerance: int,
            tokenizer: TokenizerSpec,
            batch_dim_index: int = 0,
            ctc_decode: bool = True,
            log_prediction: bool = True,
            dist_sync_on_step: bool = False,
            eou_token="\u00a5",
            sub_token_id=0
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.tolerance = tolerance
        self.tokenizer = tokenizer
        self.batch_dim_index = batch_dim_index
        self.blank_id = tokenizer.tokenizer.vocab_size
        self.ctc_decode = ctc_decode
        self.log_prediction = log_prediction

        self.eou_token = eou_token

        self.vocabulary = tokenizer.tokenizer.get_vocab()
        for char in self.vocabulary:
            if self.vocabulary[char] == sub_token_id:
                self.sub_token = char
                break

        if not hasattr(self, "sub_token"):
            raise AttributeError(f"No token with ID {sub_token_id} was found.")

        self.add_state("num_expected", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("num_predicted", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("correct_from_expected", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("correct_from_predicted", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

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

    def ctc_decoder_predictions_tensor(
        self, predictions: torch.Tensor, predictions_len: torch.Tensor = None, return_hypotheses: bool = False
    ) -> List[str]:
        """
        Decodes a sequence of labels to words

        Args:
            predictions: A torch.Tensor of shape [Batch, Time] of integer indices that correspond
                to the index of some character in the vocabulary of the tokenizer.
            predictions_len: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.
            return_hypotheses: Bool flag whether to return just the decoding predictions of the model
                or a Hypothesis object that holds information such as the decoded `text`,
                the `alignment` of emited by the CTC Model, and the `length` of the sequence (if available).
                May also contain the log-probabilities of the decoder (if this method is called via
                transcribe()) inside `y_sequence`, otherwise it is set None as it is a duplicate of
                `alignments`.

        Returns:
            Either a list of str which represent the CTC decoded strings per sample,
            or a list of Hypothesis objects containing additional information.
        """
        hypotheses = []
        # Drop predictions to CPU
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]
            # CTC decoding procedure
            decoded_prediction = []
            previous = self.blank_id
            for p in prediction:
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                previous = p

            text = self.decode_tokens_to_str(decoded_prediction)

            if not return_hypotheses:
                hypothesis = text
            else:
                hypothesis = Hypothesis(
                    y_sequence=None,  # logprob info added by transcribe method
                    score=-1.0,
                    text=text,
                    alignments=prediction,
                    length=predictions_len[ind] if predictions_len is not None else 0,
                )
            hypotheses.append(hypothesis)
        return hypotheses

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictions_lengths: torch.Tensor = None,
    ):
        num_expected = 0.0
        num_predicted = 0.0
        correct_from_expected = 0.0
        correct_from_predicted = 0.0
        distance = 0.0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[self.batch_dim_index]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decode_tokens_to_str(target)
                references.append(reference)
            if self.ctc_decode:
                hypotheses = self.ctc_decoder_predictions_tensor(predictions, predictions_lengths)
            else:
                raise NotImplementedError("Implement me if you need non-CTC decode on predictions")

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

        for h, r in zip(hypotheses, references):
            h_list = h.split()
            r_list = r.split()
            hyp_eos = set(i for i in range(len(h_list)) if h_list[i] == self.eou_token)
            ref_eos = set(i for i in range(len(r_list)) if r_list[i] == self.eou_token)
            num_expected_eou_present, total_num_expected_eou = self.compute_recall(hyp_eos, ref_eos)
            num_predicted_eou_valid, total_num_predicted_eou = self.compute_precision(hyp_eos, ref_eos)
            distance_from_expected = self.compute_distance(hyp_eos, ref_eos)

            correct_from_expected += num_expected_eou_present
            num_expected += total_num_expected_eou
            correct_from_predicted += num_predicted_eou_valid
            num_predicted += total_num_predicted_eou
            distance += distance_from_expected

        self.correct_from_expected = torch.tensor(correct_from_expected, device=self.correct_from_expected.device,
                                                  dtype=self.scores.dtype)
        self.correct_from_predicted = torch.tensor(correct_from_predicted, device=self.correct_from_predicted.device,
                                                   dtype=self.scores.dtype)
        self.num_expected = torch.tensor(num_expected, device=self.num_expected.device, dtype=self.scores.dtype)
        self.num_predicted = torch.tensor(num_predicted, device=self.num_predicted.device, dtype=self.scores.dtype)
        self.distance = torch.tensor(distance, device=self.distance.device, dtype=self.distance.dtype)

    def compute_recall(self, hyp_eos_idx: Set[int], ref_eos_idx: Set[int]) -> Tuple[float, float]:
        """
        Return the number of correct EOU token detected in hyp_eos_idx, if any. Also, return the number of expected
        EOU tokens.

        For each of the expected EOU token in ref_eos_idx, if there exists an EOU token in hyp_eos_idx with index
        within ±self.tolerance of the index of this expected EOU token, then the prediction is deemed correct.
        Otherwise, it is incorrect.
        """
        correct = 0.0
        for expected_idx in ref_eos_idx:
            for tol_level in (1, self.tolerance + 1):
                if expected_idx + tol_level in hyp_eos_idx or expected_idx - tol_level in hyp_eos_idx:
                    correct += 1.0
                    break

        return correct, len(ref_eos_idx)

    def compute_precision(self, hyp_eos_idx: Set[int], ref_eos_idx: Set[int]) -> Tuple[float, float]:
        """
        Amongst the predicted EOS tokens, return the number of correct EOU token predicted. Also, return the number of
        predicted EOU tokens.

        For each of predicted EOU token in hyp_eos_idx, if there exists an EOU token in ref_eos_token with index
        within ±self.tolerance of the index of this predicted token, then the prediction is deemed correct.
        Otherwise, it is incorrect.
        """
        correct = 0.0
        for predicted_idx in hyp_eos_idx:
            for tol_level in (1, self.tolerance + 1):
                if predicted_idx + tol_level in ref_eos_idx or predicted_idx - tol_level in ref_eos_idx:
                    correct += 1.0
                    break

        return correct, len(hyp_eos_idx)

    def compute_distance(self, hyp_eos_idx: Set[int], ref_eos_idx: Set[int]) -> float:
        """
        Compute the distance between the predicted EOU tokens within acceptable index range and the corresponding
        expected EOU token.

        """
        distance = 0.0
        for expected_idx in ref_eos_idx:
            for tol_level in (1, self.tolerance + 1):
                if expected_idx + tol_level in hyp_eos_idx or expected_idx - tol_level in hyp_eos_idx:
                    distance += tol_level
                    break

        return distance

    def compute(self) -> Tuple[float, float, float]:
        """
        Computes and returns, in the following order: recall, predcision, latency
        """
        return self.correct_from_expected / self.num_expected, \
            self.correct_from_predicted / self.num_predicted, \
            self.distance / self.correct_from_expected
