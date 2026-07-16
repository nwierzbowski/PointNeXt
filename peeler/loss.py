"""Peeler loss - Aligned Pull-Push clustering loss with curriculum ramp.

Loss = w_pull * PullLoss + w_push * PushLoss

PullLoss: Packs same-asset fragments within delta_var of centroid
PushLoss: Separates cluster centers by at least delta_dist

Curriculum: delta_var ramps down (loose→tight), delta_dist ramps up (small→large)
"""
import torch
import torch.nn as nn

from openpoints.loss.build import LOSS


@LOSS.register_module()
class AlignedPullPushPeelerLoss(nn.Module):
    """Peeler loss with curriculum ramped pull-push clustering loss."""

    def __init__(self, pull_start_delta, pull_end_delta,
                 push_start_delta, push_end_delta,
                 w_pull, w_push, **kwargs):
        super().__init__()
        self.pull_start_delta = pull_start_delta
        self.pull_end_delta = pull_end_delta
        self.push_start_delta = push_start_delta
        self.push_end_delta = push_end_delta
        self.w_pull = w_pull
        self.w_push = w_push

    def forward(self, gcn_embeddings, asset_ids, mask, epoch, num_epochs):
        """
        Args:
            gcn_embeddings: (B, N, D) - refined embeddings
            asset_ids: (B, N) - integer group IDs (-1 for padding)
            mask: (B, N) - 1 for real fragments, 0 for padding
            epoch: current epoch number (int)
            num_epochs: total target epochs (int)

        Returns:
            loss: scalar total loss
            loss_dict: dict with loss components
        """
        progress = (epoch - 1) / max(num_epochs - 1, 1)  # 0→1 over training

        # Linear curriculum ramp
        delta_var = self.pull_start_delta + progress * (self.pull_end_delta - self.pull_start_delta)
        delta_dist = self.push_start_delta + progress * (self.push_end_delta - self.push_start_delta)

        pull_loss, push_loss = self._aligned_pull_push_loss(gcn_embeddings, asset_ids, mask, delta_var, delta_dist)

        total = self.w_pull * pull_loss + self.w_push * push_loss
        return total, {
            'loss_total': total.item(),
            'loss_pull': pull_loss.item(),
            'loss_push': push_loss.item(),
        }

    def _aligned_pull_push_loss(self, embeddings, group_ids, mask, delta_var, delta_dist):
        """Vectorized Aligned Pull-Push loss geometrically aligned for DBSCAN eps=0.5.

        Pull: Forces same-asset fragments within delta_var of centroid
        Push: Forces different-asset centroids at least delta_dist apart

        Args:
            embeddings: (B, N, D) - refined embeddings
            group_ids: (B, N) - integer group IDs (-1 for padding)
            mask: (B, N) - 1 for real fragments
            delta_var: pull margin (ramps down during training)
            delta_dist: push margin (ramps up during training)

        Returns:
            tuple: (pull_loss, push_loss)
        """
        B, N, D = embeddings.shape
        device = embeddings.device

        # Create valid mask: 1 for real fragments, 0 for padding
        valid_mask = (mask > 0.5) & (group_ids >= 0)  # (B, N)

        pull_loss_sum = torch.tensor(0.0, device=device)
        push_loss_sum = torch.tensor(0.0, device=device)
        valid_batches = 0

        for b in range(B):
            v_mask = valid_mask[b]  # (N,)
            if v_mask.sum() == 0:
                continue

            b_ids = group_ids[b][v_mask]  # (N_valid,)
            b_embs = embeddings[b][v_mask]  # (N_valid, D)

            # Get unique group IDs
            unique_groups = torch.unique(b_ids)
            if len(unique_groups) == 0:
                continue

            # Map original group IDs to contiguous indices (0, 1, 2, ...)
            group_to_idx = {gid.item(): i for i, gid in enumerate(unique_groups)}
            reindexed = torch.tensor([group_to_idx[gid.item()] for gid in b_ids], device=device)

            # Compute centroids using scatter_mean
            centroids = torch.zeros(len(unique_groups), D, device=device)
            counts = torch.zeros(len(unique_groups), device=device)
            centroids.index_add_(0, reindexed, b_embs)
            counts.index_add_(0, reindexed, torch.ones(len(reindexed), dtype=embeddings.dtype, device=embeddings.device))
            counts = torch.clamp(counts, min=1)
            centroids = centroids / counts.unsqueeze(-1)

            # PULL LOSS: distance from each point to its centroid
            centroid_indices = reindexed  # (N_valid,)
            batch_centroids = centroids[centroid_indices]  # (N_valid, D)
            dists = torch.norm(b_embs - batch_centroids, dim=-1)  # (N_valid,)
            pull = torch.clamp(dists - delta_var, min=0.0) ** 2
            pull_loss_sum = pull_loss_sum + pull.mean()

            # PUSH LOSS: pairwise centroid distances
            num_groups = len(unique_groups)
            if num_groups > 1:
                centroid_dists = torch.cdist(centroids, centroids)  # (C, C)
                triu_indices = torch.triu_indices(num_groups, num_groups, offset=1)
                pair_dists = centroid_dists[triu_indices[0], triu_indices[1]]  # (C*(C-1)/2,)
                push = torch.clamp(delta_dist - pair_dists, min=0.0) ** 2
                push_loss_sum = push_loss_sum + push.mean()

            valid_batches += 1

        # Average over batches
        if valid_batches > 0:
            pull_loss = pull_loss_sum / valid_batches
            push_loss = push_loss_sum / valid_batches
        else:
            pull_loss = torch.tensor(0.0, device=device)
            push_loss = torch.tensor(0.0, device=device)

        return pull_loss, push_loss



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

    def forward(self, affinity_logits, Y, mask, epoch=None, num_epochs=None):
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
        bce_per_element = self.bce(affinity_logits, Y * 0.9 + 0.05)  # (B, N, N)

        # 2. Compute dynamic class weights PER-SAMPLE (keep batch dim as B, 1, 1)
        # sum over spatial dimensions (1, 2) but preserve the batch dimension
        n_pos = (Y * valid_mask).sum(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        total_active = valid_mask.sum(dim=(1, 2), keepdim=True) # (B, 1, 1)
        n_neg = total_active - n_pos                            # (B, 1, 1)
        
        eps = 1e-6
        raw_pos_weight = total_active / (2.0 * n_pos + eps)
        raw_neg_weight = total_active / (2.0 * n_neg + eps)
        
        # Clamp to prevent extreme gradient updates on highly imbalanced samples
        dynamic_pos_weight = torch.clamp(raw_pos_weight, min=0.1, max=25.0)
        dynamic_neg_weight = torch.clamp(raw_neg_weight, min=0.1, max=25.0)
        
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