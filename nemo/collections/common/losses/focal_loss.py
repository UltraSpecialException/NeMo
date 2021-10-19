import onnx.backend.test.case.node.negativeloglikelihoodloss
import torch
import torch.nn as nn
from typing import Optional
from nemo.utils import logging


class FocalLoss(nn.NLLLoss):
    """
    Implements Focal Loss as presented in the paper Focal Loss for Dense Object Detection.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__(weight=alpha, reduction="none")
        self.gamma = gamma
        self.focal_loss_reduction = reduction

    def forward(self, logits, labels, loss_mask=None):
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        log_probs = logits_flatten.log_softmax(dim=-1)
        probs = torch.exp(log_probs)

        non_reduced_loss = super().forward((1 - probs) ** self.gamma * log_probs, labels_flatten)

        if self.focal_loss_reduction == "mean":
            return non_reduced_loss.mean()
        else:
            raise NotImplementedError(f"{self.reduction} is not a valid reduction strategy.")
