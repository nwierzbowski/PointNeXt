"""Peeler loss - Aligned Pull-Push clustering loss with curriculum ramp.

Loss = w_pull * PullLoss + w_push * PushLoss

PullLoss: Packs same-asset fragments within delta_var of centroid
PushLoss: Separates cluster centers by at least delta_dist

Curriculum: delta_var ramps down (loose→tight), delta_dist ramps up (small→large)
"""
import torch
import torch.nn as nn

from openpoints.loss.build import LOSS
import torch.nn.functional as F

@LOSS.register_module()
class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss for Associative Embedding.
    
    Pulls same-asset fragment embeddings together and pushes different-asset
    fragment embeddings apart on a unit hypersphere.
    """
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, Y, mask):
        """
        Args:
            embeddings: (B, N, D_embed) - Node embeddings
            Y: (B, N, N) - Ground truth same asset pairwise mask (1 for same, 0 otherwise)
            mask: (B, N) - 1 for real fragments, 0 for padding
        """
        B, N, D = embeddings.shape
        device = embeddings.device

        # 1. Enforce L2-normalization to guarantee embeddings are on the unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # 2. Compute cosine similarity matrix: (B, N, N)
        similarity_matrix = torch.bmm(embeddings, embeddings.transpose(1, 2)) / self.temperature

        # 3. Generate boolean valid pair mask (excludes diagonal and padding elements)
        valid_mask = mask.unsqueeze(1).bool() & mask.unsqueeze(2).bool()
        diag_idx = torch.arange(N, device=device)
        valid_mask[:, diag_idx, diag_idx] = False  # Excludes self-contrast

        # 4. FP16/AMP Friendly Masking
        # -1e4 is safe for float16 (max limit -65504) and guarantees exp(-1e4) -> 0
        safe_min = -1e4 if embeddings.dtype == torch.float16 else -1e9
        
        # Compute Log-Sum-Exp safely
        log_sum_exp = torch.logsumexp(similarity_matrix.masked_fill(~valid_mask, safe_min), dim=-1, keepdim=True)

        # Guard against fully masked rows (e.g., padded nodes) to prevent NaN when computing log_prob
        # If a row has no valid neighbors, log_sum_exp would be safe_min. We zero it out here.
        has_valid = valid_mask.any(dim=-1, keepdim=True)
        log_sum_exp = torch.where(has_valid, log_sum_exp, torch.zeros_like(log_sum_exp))

        # 5. Compute stable log-probabilities
        log_prob = similarity_matrix - log_sum_exp

        # 6. Extract only valid positive (same-asset) pairs
        pos_mask = Y.bool() & valid_mask

        # Identify active nodes that actually have positive partners
        has_positives = (pos_mask.sum(dim=-1) > 0)
        active_nodes_with_pos = (mask.bool() & has_positives).float()

        # 7. Compute mean log-probability over positive pairs per node
        sum_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=-1)
        mean_log_prob_pos = sum_log_prob_pos / (pos_mask.sum(dim=-1).float() + 1e-8)

        # 8. Global normalization to prevent loss dilution across varying batch sizes
        loss_per_node = -mean_log_prob_pos * active_nodes_with_pos
        
        total_active_nodes = active_nodes_with_pos.sum()
        loss = loss_per_node.sum() / (total_active_nodes + 1e-8)

        return loss, {
            'loss_total': loss.detach()
        }


@LOSS.register_module()
class BCEAffinityPeelerLoss(nn.Module):
    """BCE loss on pairwise affinity logits for purely relational peeler.

    Computes binary cross-entropy between predicted affinity logits and
    ground truth same-asset matrix Y, masked to exclude padding pairs
    and self-pairs. Dynamically balances positive/negative classes per-sample.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, affinity_logits, Y, mask):
        """
        Args:
            affinity_logits: (B, N, N) - predicted affinity logits
            Y: (B, N, N) - ground truth same-asset matrix (0/1)
            mask: (B, N) - 1 for real fragments, 0 for padding
            epoch: current epoch number (optional)
            num_epochs: total target epochs (optional)

        Returns:
            loss: scalar total loss
            loss_dict: dict with loss components
        """
        valid = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        N = Y.shape[1]
        diag_mask = 1.0 - torch.eye(N, device=Y.device).unsqueeze(0)  # (1, N, N)
        valid_mask = valid * diag_mask  # (B, N, N)

        # 1. Compute raw, unweighted per-element BCE
        bce_per_element = self.bce(affinity_logits, Y)  # (B, N, N)

        # 2. Compute dynamic class weights PER-SAMPLE (keep batch dim as B, 1, 1)
        # sum over spatial dimensions (1, 2) but preserve the batch dimension
        n_pos = (Y * valid_mask).sum(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        total_active = valid_mask.sum(dim=(1, 2), keepdim=True) # (B, 1, 1)
        n_neg = total_active - n_pos                            # (B, 1, 1)
        
        eps = 1e-6
        raw_pos_weight = total_active / (2.0 * n_pos + eps)
        raw_neg_weight = total_active / (2.0 * n_neg + eps)
        # print("raw_pos_weight", torch.max(raw_pos_weight), torch.min(raw_pos_weight))
        # print("raw_neg_weight", torch.max(raw_neg_weight), torch.min(raw_neg_weight))
        
        # Clamp to prevent extreme gradient updates on highly imbalanced samples
        dynamic_pos_weight = torch.clamp(raw_pos_weight, min=0.1, max=500)
        dynamic_neg_weight = torch.clamp(raw_neg_weight, min=0.1, max=500)
        
        # Construct the weight matrix
        pos_w = torch.where(Y > 0.5, dynamic_pos_weight, dynamic_neg_weight)  # (B, N, N)
        
        # Apply weights and mask
        weighted_bce = bce_per_element * pos_w * valid_mask  # (B, N, N)

        # 3. Safe per-sample reduction (only average over samples that have active elements)
        sample_sums = weighted_bce.sum(dim=(1, 2))                     # (B,)
        sample_counts = valid_mask.sum(dim=(1, 2))                     # (B,)
        
        active_samples_mask = sample_counts > 0
        loss = (sample_sums[active_samples_mask] / sample_counts[active_samples_mask]).mean()

        # 4. Separate raw, unweighted pos/neg BCE for clean logging
        pos_mask = (Y > 0.5) & (valid_mask > 0)
        neg_mask = (Y <= 0.5) & (valid_mask > 0)
        
        # Use bce_per_element instead of weighted_bce to get clean, interpretable metrics
        pos_loss = bce_per_element[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=affinity_logits.device)
        neg_loss = bce_per_element[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=affinity_logits.device)

        return loss, {
            'loss_total': loss.item(),
            'loss_pos': pos_loss.item(),
            'loss_neg': neg_loss.item(),
        }



@LOSS.register_module()
class FocalAffinityPeelerLoss(nn.Module):
    """Alpha-Balanced Focal Loss on pairwise affinity logits.

    Uses a fixed alpha constant for class balancing and the focal modulating
    factor (1 - p_t)^gamma to silence easy negatives.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, affinity_logits, Y, mask):
        """
        Args:
            affinity_logits: (B, N, N) - predicted affinity logits
            Y: (B, N, N) - ground truth same-asset matrix (0/1)
            mask: (B, N) - 1 for real fragments, 0 for padding
        """
        valid = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        N = Y.shape[1]
        diag_mask = 1.0 - torch.eye(N, device=Y.device).unsqueeze(0)  # (1, N, N)
        valid_mask = valid * diag_mask  # (B, N, N)

        # 1. Compute raw, unweighted per-element BCE (no label smoothing needed!)
        bce_per_element = self.bce(affinity_logits, Y)  # (B, N, N)

        # 2. Compute the Focal Loss modulating factor: (1 - p_t)^gamma
        probs = torch.sigmoid(affinity_logits)
        p_t = torch.where(Y > 0.5, probs, 1.0 - probs)
        modulating_factor = (1.0 - p_t) ** self.gamma

        # 3. Apply fixed alpha-balancing (0.75 for positives, 0.25 for negatives)
        alpha_t = torch.where(Y > 0.5, self.alpha, 1.0 - self.alpha)

        # 4. Apply Focal modulation, alpha-balancing, and masks
        focal_bce = bce_per_element * modulating_factor * alpha_t * valid_mask  # (B, N, N)

        # 5. Safe per-sample reduction (average over active elements)
        sample_sums = focal_bce.sum(dim=(1, 2))                     # (B,)
        sample_counts = valid_mask.sum(dim=(1, 2))                     # (B,)
        
        active_samples_mask = sample_counts > 0
        loss = (sample_sums[active_samples_mask] / sample_counts[active_samples_mask]).mean()

        # 6. Separate raw, unweighted pos/neg BCE for clean logging
        pos_mask = (Y > 0.5) & (valid_mask > 0)
        neg_mask = (Y <= 0.5) & (valid_mask > 0)
        
        pos_loss = bce_per_element[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=affinity_logits.device)
        neg_loss = bce_per_element[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=affinity_logits.device)

        return loss, {
            'loss_total': loss.item(),
            'loss_pos': pos_loss.item(),
            'loss_neg': neg_loss.item(),
        }