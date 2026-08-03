"""Peeler loss - Cluster-Equalized Focal Loss with explicit mask tensors.

No sentinel magic numbers. Model outputs raw logits, loss uses explicit masks.
"""
import torch
import torch.nn as nn

from openpoints.loss.build import LOSS


@LOSS.register_module()
class ClusterFocalPeelerLoss(nn.Module):
    """Cluster-Equalized Focal Loss for BOTH positive and negative pairs,
    with Triplet Transitivity Penalty.

    Positives: Equalized per-cluster (equal weight per asset).
    Negatives: Equalized per-cluster (equal weight to isolating every asset).
    Transitivity: Soft-AND triplet penalty to eliminate bridge edges.

    Uses explicit boolean/float masks (no sentinel magic numbers).
    """
    def __init__(self, gamma: float = 2.0,
                 pos_weight: float = 1.0, neg_weight: float = 2.0,
                 transitivity_weight: float = 0.05, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.transitivity_weight = transitivity_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def _scene_loss(self, logits, scene_asset_ids, scene_mask=None):
        """Compute focal loss for a single scene matrix (N_s, N_s)."""
        N = logits.shape[0]
        device = logits.device

        same_cluster = (scene_asset_ids.unsqueeze(1) == scene_asset_ids.unsqueeze(0)).float()
        diag_mask = 1.0 - torch.eye(N, device=device)

        # Explicit masking: combine valid fragment mask with non-self diagonal mask
        if scene_mask is not None:
            valid_mask = (scene_mask.unsqueeze(1) * scene_mask.unsqueeze(0)) * diag_mask
        else:
            valid_mask = diag_mask

        bce_per_element = self.bce(logits, same_cluster * 0.9 + 0.05)
        probs = torch.sigmoid(logits)
        p_t = torch.where(same_cluster > 0.5, probs, 1.0 - probs)
        focal_per_pair = bce_per_element * ((1.0 - p_t) ** self.gamma)

        cluster_offset = max(0, -int(scene_asset_ids.min().item()))
        safe_ids = (scene_asset_ids + cluster_offset).long()
        max_cluster = int(scene_asset_ids.max().item()) + cluster_offset + 2

        # === 1. CLUSTER-EQUALIZED POSITIVE LOSS ===
        pos_valid = valid_mask * same_cluster
        pos_focal = focal_per_pair * pos_valid
        pos_flat = pos_focal.view(-1)
        pair_cluster = safe_ids.unsqueeze(1).expand(N, N).reshape(-1)

        cluster_pos_sum = torch.zeros(max_cluster, device=device)
        cluster_pos_count = torch.zeros(max_cluster, device=device)
        cluster_pos_sum.scatter_add_(0, pair_cluster, pos_flat)
        cluster_pos_count.scatter_add_(0, pair_cluster, pos_valid.view(-1))
        cluster_pos_mean = (cluster_pos_sum / (cluster_pos_count + 1e-8)).masked_fill(cluster_pos_count == 0, 0.0)
        active_pos = (cluster_pos_count > 0).sum().float()
        pos_loss = cluster_pos_mean.sum() / (active_pos + 1e-8) if active_pos > 0 else torch.tensor(0.0, device=device)

        # === 2. CLUSTER-EQUALIZED NEGATIVE LOSS ===
        neg_valid = valid_mask * (1.0 - same_cluster)
        neg_focal = focal_per_pair * neg_valid
        node_neg_focal = neg_focal.sum(dim=1)
        node_neg_count = neg_valid.sum(dim=1)

        cluster_neg_sum = torch.zeros(max_cluster, device=device)
        cluster_neg_count = torch.zeros(max_cluster, device=device)
        cluster_neg_sum.scatter_add_(0, safe_ids, node_neg_focal)
        cluster_neg_count.scatter_add_(0, safe_ids, node_neg_count)
        cluster_neg_mean = (cluster_neg_sum / (cluster_neg_count + 1e-8)).masked_fill(cluster_neg_count == 0, 0.0)
        active_neg = (cluster_neg_count > 0).sum().float()
        neg_loss = cluster_neg_mean.sum() / (active_neg + 1e-8) if active_neg > 0 else torch.tensor(0.0, device=device)

        # === 3. TRIPLET TRANSITIVITY LOSS ===
        trans_loss = torch.tensor(0.0, device=device)
        if self.transitivity_weight > 0.0 and N >= 3:
            valid_ij = (valid_mask > 0).unsqueeze(2)  # (N, N, 1) -> (i, j)
            valid_jk = (valid_mask > 0).unsqueeze(0)  # (1, N, N) -> (j, k)
            valid_ik = (valid_mask > 0).unsqueeze(1)  # (N, 1, N) -> (i, k)
            valid_triplet = valid_ij & valid_jk & valid_ik  # (N, N, N)

            A_ij = probs.unsqueeze(2)
            A_jk = probs.unsqueeze(0)
            A_ik = probs.unsqueeze(1)

            violation = torch.relu(A_ij * A_jk - A_ik)
            trans_bce = (violation ** 2) * valid_triplet.float()
            trans_sum = trans_bce.sum()
            trans_count = valid_triplet.float().sum()
            if trans_count > 0:
                trans_loss = trans_sum / trans_count

        scene_total = self.pos_weight * pos_loss + self.neg_weight * neg_loss + self.transitivity_weight * trans_loss
        return scene_total, pos_loss.item(), neg_loss.item(), trans_loss.item()

    def forward(self, scene_logits, asset_ids, scene_ids=None, mask=None):
        """Compute loss over per-scene logits.

        Args:
            scene_logits: Dict[int, (N_s, N_s)] or single (N, N) tensor
            asset_ids: (N_total,) flat asset IDs
            scene_ids: (N_total,) flat scene IDs, or None for single scene
            mask: (N_total,) optional fragment validity mask
        """
        if isinstance(scene_logits, dict):
            scenes = scene_logits
        else:
            scenes = {0: scene_logits}

        total_loss = torch.tensor(0.0, device=next(iter(scenes.values())).device)
        sum_pos, sum_neg, sum_trans = 0.0, 0.0, 0.0
        count = 0

        for scene_idx, logits in scenes.items():
            if scene_ids is not None:
                scene_mask_idx = (scene_ids == scene_idx)
                scene_asset_ids = asset_ids[scene_mask_idx]
                scene_m = mask[scene_mask_idx] if mask is not None else None
            else:
                scene_asset_ids = asset_ids
                scene_m = mask

            scene_l, p, n, t = self._scene_loss(logits, scene_asset_ids, scene_mask=scene_m)
            total_loss = total_loss + scene_l
            sum_pos += p
            sum_neg += n
            sum_trans += t
            count += 1

        total_loss = total_loss / count
        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_pos': sum_pos / count,
            'loss_neg': sum_neg / count,
            'loss_trans': sum_trans / count,
        }
