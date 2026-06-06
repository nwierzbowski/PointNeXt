"""Peeler loss - expected BCE over anchor distribution.

Loss = E_{i ~ P_anchor}[BCE(membership_logits_i, Y_i)]

For each anchor i, compute BCE against ground truth for all candidate pairs.
Weight each anchor's error by its probability P_anchor[i].

This replaces the old argmax-based loss with a smooth, differentiable objective
that lets the anchor head learn from all anchors, not just the top-1.
"""
import torch.nn.functional as F
import torch
import torch.nn as nn

from openpoints.loss.build import LOSS


@LOSS.register_module()
class PeelerLoss(nn.Module):
    """Expected BCE loss over anchor distribution.

    Loss = mean_b( sum_i( P_anchor[i,b] * mean_j( BCE(affinity_logits[i,j], Y[i,j]) * mask[i] * mask[j] ) ) / mean(mask) )
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, anchor_probs, affinity_logits, Y, mask):
        """
        Args:
            anchor_probs: (B, N) - softmax distribution over anchors
            affinity_logits: (B, N, N) - raw logits for ALL anchor-candidate pairs
            Y: (B, N, N) - ground truth same-asset matrix
            mask: (B, N) - 1 for real fragments, 0 for padding

        Returns:
            loss: scalar expected BCE loss
            loss_dict: dict with loss components
        """
        B, N, _ = Y.shape

        # 1. Calculate BCE for EVERY possible relationship in the soup
        bce_matrix = F.binary_cross_entropy_with_logits(affinity_logits, Y, reduction='none')  # [B, N, N]

        # 2. Mask the padding (Both rows and columns)
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]
        masked_bce = bce_matrix * mask_2d  # [B, N, N]

        # 3. Calculate the error per anchor (mean across candidates)
        error_per_anchor = masked_bce.sum(dim=2) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # [B, N]

        # 4. THE EXPECTED LOSS: Weight errors by anchor probabilities
        expected_loss = (anchor_probs * error_per_anchor).sum(dim=1).mean()  # scalar

        return expected_loss, {
            'loss_total': expected_loss.item(),
            'loss_bce': expected_loss.item(),
        }
