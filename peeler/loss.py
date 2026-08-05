"""Peeler loss - Sparse Focal BCE per candidate edge.

Each of the K candidates is an independent binary classifier:
    p_k = sigmoid(L_k)

Focal BCE down-weights easy negatives (non-matching candidates).
Handles both "no match" (all logits negative) and "multi-match" (multiple positives).
"""
import torch
import torch.nn as nn

from openpoints.loss.build import LOSS


@LOSS.register_module()
class SparseFocalBCELoss(nn.Module):
    """Sparse Focal BCE loss for Top-K candidate edges.

    Per-candidate focal loss:
        p = sigmoid(logit_k)
        p_t = p * y + (1-p) * (1-y)
        focal_weight = (1 - p_t) ^ gamma
        loss_k = -alpha * y * log(p) * focal_weight - (1-alpha) * (1-y) * log(1-p) * focal_weight

    Args:
        gamma: focal loss gamma (default 2.0). Down-weights easy examples.
        alpha: positive class weight (default 0.5). Negative weight = 1-alpha.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, indices, asset_ids, candidate_mask):
        """Compute focal BCE loss.

        Args:
            logits: (N, K) raw logits
            indices: (N, K) global candidate indices
            asset_ids: (N_total,) tensor of asset IDs for label derivation
            candidate_mask: (N, K) candidate validity mask

        Returns:
            (loss, metrics_dict)
        """
        # Vectorized label generation using global indices
        seed_assets = asset_ids.unsqueeze(-1)  # (N, 1)
        cand_assets = asset_ids[indices]       # (N, K)
        labels = (seed_assets == cand_assets).float()  # (N, K)

        # Sigmoid probabilities
        probs = torch.sigmoid(logits)

        # Focal BCE per candidate
        p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
        focal_weight = (1.0 - p_t) ** self.gamma

        bce = -self.alpha * labels * torch.log(probs.clamp(min=1e-8)) \
              - (1.0 - self.alpha) * (1.0 - labels) * torch.log((1.0 - probs).clamp(min=1e-8))
        focal_per_candidate = bce * focal_weight


        # Count positives and negatives
        n_pos = (labels > 0.5).logical_and(candidate_mask).sum().item()
        n_neg = (labels <= 0.5).logical_and(candidate_mask).sum().item()

        # Sum and normalize
        focal_masked = focal_per_candidate * candidate_mask.float()
        n_valid = candidate_mask.sum().item()
        loss = focal_masked.sum() / (n_valid + 1e-8)

        return loss, {
            'loss_total': loss.item(),
            'n_pos': n_pos,
            'n_neg': n_neg,
        }
