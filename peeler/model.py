"""Peeler - Attention backbone with relational bias for fragment clustering.

Architecture:
    PeelerBackbone: embedding projection → self-attention with pairwise relational bias
    PurelyRelationalBlock: bottleneck triangular update with fixed highway
    
    Peeler: backbone + projection for clustering
    PurelyRelationalPeeler: affinity logits output for BCE training

Training: Aligned Pull-Push clustering loss with curriculum ramp.
Optimization: End-to-end training of attention backbone.

Registered with openpoints MODELS registry.
"""
import torch
import torch.nn as nn

from openpoints.models.build import MODELS


TRANSFORM_DIM = 6
_REL_FEATURE_DIM = 22


def transforms_to_pose(transforms, mask):
    """Extract translation, scale, and normalized rotation from 4x4 transforms.

    Args:
        transforms: (B, N, 16) f32 - row-major 4x4 matrices
        mask: (B, N) optional mask for purely relational backbone API compat

    Returns:
        translation: (B, N, 3) normalized by average distance
        scale: (B, N, 1) clamped to min 1e-8, normalized by average distance
        rot: (B, N, 9) flattened normalized rotation matrix
    """
    B, N, _ = transforms.shape
    mat = transforms.view(B, N, 4, 4)
    translation = mat[:, :, :3, 3]
    scale = torch.norm(mat[:, :, :3, :3], dim=-1).mean(-1, keepdim=True)
    scale = torch.clamp(scale, min=1e-8)
    rot = mat[:, :, :3, :3].reshape(B, N, -1) / scale

    mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
    num_active = mask_expanded.sum(dim=1, keepdim=True)  # (B, 1, 1)

    active_translation = translation * mask_expanded
    center = active_translation.sum(dim=1, keepdim=True) / (num_active + 1e-8)

    relative_translation = (translation - center) * mask_expanded

    # Normalize by average distance per batch
    active_scales_sum = (scale * mask_expanded).sum(dim=1, keepdim=True)  # (B, 1, 1)
    avg_distance = active_scales_sum / (num_active + 1e-8)  # (B, 1, 1)

    # active_distance_sum = torch.sum(torch.norm(relative_translation, dim=-1), dim=-1)
    # avg_distance = active_distance_sum / (num_active.squeeze() + 1e-8)  # (B,) / (B,) = (B,)
    # avg_distance = torch.clamp(avg_distance, min=1e-8)
    # avg_distance = avg_distance.view(B, 1, 1)  # (B, 1, 1)

    # Break degeneracy: when all positions are identical, add small deterministic
    # per-fragment perturbation based on fragment index. ONNX-compatible (no randomness).
    indices = torch.arange(N, device=transforms.device, dtype=torch.float32)
    perturbation = torch.sin(indices.unsqueeze(-1) * 0.1) * 1e-4  # (N, 3)
    relative_translation = relative_translation + perturbation.unsqueeze(0)  # broadcast to (B, N, 3)
    
    relative_translation = relative_translation / avg_distance
    scale = scale / avg_distance

    return relative_translation, scale, rot


def compute_transform_features(translation, scale):
    """Compute transform features for anchor scoring.

    Args:
        translation: (B, N, 3) normalized translation
        scale: (B, N, 1) normalized scale

    Returns:
        features: (B, N, 6) concatenated transform features
    """
    distance = torch.norm(translation, dim=-1, keepdim=True)
    norm_dist = torch.log10(torch.clamp(distance / scale, min=1e-8)) / 7
    dir = torch.where(distance > 1e-8, translation / distance, torch.zeros_like(translation))
    distance = (torch.log10(torch.clamp(distance, min=1e-3)) + 3) / 4
    norm_scale = (torch.log10(scale)) / 6
    return torch.cat([distance, norm_dist, dir, norm_scale], dim=-1)


def compute_relative_features(seed_T, seed_S, cand_T, cand_S, seed_rot, cand_rot):
    """Compute relative features between seed and candidate poses.

    Args:
        seed_T: (B, S, 3) f32 - seed translations
        seed_S: (B, S) f32 - seed scales
        cand_T: (B, N, 3) f32 - candidate translations
        cand_S: (B, N) f32 - candidate scales
        seed_rot: (B, S, 9) f32 - seed rotation matrices (flattened)
        cand_rot: (B, N, 9) f32 - candidate rotation matrices (flattened)

    Returns:
        rel_features: (B, S, N, 22) f32 - includes rotation features
    """
    B, S, _ = seed_T.shape
    _, N, _ = cand_T.shape
    device = seed_T.device

    diff = cand_T.unsqueeze(1) - seed_T.unsqueeze(2)  # (B, S, N, 3)
    dist_raw = torch.norm(diff, dim=-1, keepdim=True)  # (B, S, N, 1)
    
    seed_S_exp = seed_S.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
    cand_S_exp = cand_S.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)

    # 1. Reshape 9D flat vectors back to 3x3 rotation matrices
    R_seed = seed_rot.reshape(B, S, 3, 3)
    R_cand = cand_rot.reshape(B, N, 3, 3)

    # 2. Unsqueeze and transpose seed rotation for batched matrix multiplication [3, 4]
    R_seed_T = R_seed.unsqueeze(2).transpose(-2, -1)  # (B, S, 1, 3, 3)
    R_cand_exp = R_cand.unsqueeze(1)                  # (B, 1, N, 3, 3)

    # 3. Compute relative rotation: [B, S, N, 3, 3]
    R_rel = torch.matmul(R_seed_T, R_cand_exp)

    # 4. Flatten the 3x3 relative rotation matrix back into a 9D vector
    # This features is now 100% rotation-invariant!
    rot_diff = R_rel.reshape(B, S, N, 9)

    # 5. Optimized Cosine Similarity using the Trace [3]
    # Trace(R_rel) = R_rel[0,0] + R_rel[1,1] + R_rel[2,2]
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2] # (B, S, N)
    rot_cosine = (trace / 3.0).unsqueeze(-1)                      # (B, S, N, 1)

    return torch.cat([
        torch.where(dist_raw > 1e-8, diff / dist_raw, torch.zeros_like(diff)),  # direction (3)
        (torch.log10(torch.clamp(dist_raw, min=1e-8)) / 8) + 1,                  # dist (1)
        (torch.log10(torch.clamp(torch.clamp(dist_raw - seed_S_exp - cand_S_exp, min=1e-8), min=1e-8)) / 8) + 1,  # dist_bwn (1)
        (torch.log10(torch.clamp((torch.clamp(dist_raw - seed_S_exp - cand_S_exp, min=1e-8)) / seed_S_exp, min=1e-8)) / 8),  # dist_bwn_normalized_s (1)
        (torch.log10(torch.clamp((torch.clamp(dist_raw - seed_S_exp - cand_S_exp, min=1e-8)) / cand_S_exp, min=1e-8)) / 8),  # dist_bwn_normalized_c (1)
        (torch.log10(torch.clamp(dist_raw / seed_S_exp, min=1e-8)) / 8),          # dist_normalized_s (1)
        (torch.log10(torch.clamp(dist_raw / cand_S_exp, min=1e-8)) / 8),          # dist_normalized_c (1)
        (torch.log10(seed_S_exp.expand(-1, -1, N, -1)) / 6),                      # seed_S_log (1)
        (torch.log10(cand_S_exp.expand(-1, S, -1, -1)) / 6),                      # cand_S_log (1)
        (torch.log10(cand_S_exp / seed_S_exp) / 8),                                # rel_scale (1)
        rot_diff,                                                                  # rot_diff (9)
        rot_cosine,                                                                # rot_cosine (1)
    ], dim=-1)


@MODELS.register_module()
class Peeler(nn.Module):
    """Peeler model with attention backbone for fragment clustering.

    Architecture:
        1. PeelerBackbone: embedding projection + self-attention with pairwise relational bias
        2. Projection: to output dimension for clustering

    Input: embeddings(B,N,feat_dim), transforms(B,N,16), mask(B,N)
    Output: refined_embeddings(B,N,gcn_out_dim)

    Training: Aligned Pull-Push clustering loss with curriculum ramp.
    Inference: embeddings clusterable with DBSCAN.
    """

    def __init__(self,
                   feat_dim,
                   attention_heads,
                   attention_blocks,
                   gcn_out_dim,
                   **kwargs):
        super().__init__()
        self.feat_dim = feat_dim

        # 1. Attention backbone: produces embeddings for clustering
        from peeler.attention_backbone import PeelerBackbone
        self.backbone = PeelerBackbone(feat_dim, feat_dim, attention_heads, attention_blocks)
        self.gcn_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, gcn_out_dim),
        )

    def forward(self, embeddings, transforms, mask):
        """Forward pass.

        Args:
            embeddings: (B, N, feat_dim) - fragment embeddings from backbone
            transforms: (B, N, 16) - fragment transforms (4x4 pose matrices flattened)
            mask: (B, N) - 1 for real fragments, 0 for padding

        Returns:
            refined_emb: (B, N, gcn_out_dim) - clusterable embeddings
        """
        B, N, _ = embeddings.shape

        # 1. Attention backbone refinement
        refined_emb = self.backbone(transforms, mask, embeddings)

        # 2. Project to output dimension
        refined_emb = self.gcn_proj(refined_emb)
        refined_emb = refined_emb * mask.unsqueeze(-1)

        return refined_emb

class GeGLU(nn.Module):
    """
    ONNX-safe Gated Linear Unit with GELU activation.
    Splits the last dimension in half and uses one half to gate the other.
    """
    def forward(self, x):
        # chunk(2, dim=-1) splits the channel dimension into two equal tensors
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.nn.functional.silu(x2)

class PurelyRelationalBlock(nn.Module):
    """Bottleneck Purely Relational Block with fixed highway.

    Projects input down to target dim, runs relational update,
    projects back up, then adds result to the residual highway.

    highway_dim: fixed width of the residual connection
    target_dim: reduced dimension for the matmul computation
    """

    def __init__(self, highway_dim, target_dim):
        super().__init__()
        
        self.left_proj = nn.Sequential(
            nn.Linear(highway_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            GeGLU(),
        )
        self.right_proj = nn.Sequential(
            nn.Linear(highway_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            GeGLU(),
        )
        self.proj_up = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, target_dim * 4),
            GeGLU(),
            nn.Linear(target_dim * 2, highway_dim)
        )

    def forward(self, e, mask):
        B, N, _, D = e.shape
        highway = e
        left = self.left_proj(e)
        right = self.right_proj(e)
        mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        left = left * mask_2d
        right = right * mask_2d
        left_perm = left.permute(0, 3, 1, 2)
        right_perm = right.permute(0, 3, 1, 2)
        triangular_sum = torch.matmul(left_perm, right_perm)
        triangular_out = triangular_sum.permute(0, 2, 3, 1)
        result = self.proj_up(triangular_out)
        result = result * mask_2d
        return highway + result


class MLPBlock(nn.Module):
    """MLP residual block on pairwise feature tensor.

    Projects down to target_dim, applies MLP, projects back up to highway_dim.
    Element-wise operations on (B, N, N, D) with bottleneck architecture.
    """

    def __init__(self, highway_dim, target_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(highway_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            GeGLU(),
            nn.Linear(target_dim, highway_dim)
        )

    def forward(self, e, mask):
        result = self.mlp(e)
        mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        result = result * mask_2d
        return e + result


_PAIRWISE_DIM = 42

@MODELS.register_module()
class PurelyRelationalPeeler(nn.Module):
    """Purely relational peeler using triangular edge updates.

    Architecture:
        1. Compute 22D relative features from transforms
        2. Compute 4D pairwise features from embeddings
        3. Concat features (26D) and project to highway_dim relational space
        4. Stack of relational + MLP blocks with bottleneck
        5. Project to 1-channel affinity logits

    Input: embeddings(B,N,256), transforms(B,N,16), mask(B,N)
    Output: affinity_logits(B,N,N)

    Training: BCE loss on same-asset affinity matrix.
    Inference: threshold + connected components for clustering.
    """

    def __init__(self, downsample_schedule, mlp_sizes, highway_dim, pairwise_dropout, **kwargs):
        super().__init__()
        self.downsample_schedule = list(downsample_schedule)
        self.num_blocks = len(self.downsample_schedule)
        self.mlp_sizes = list(mlp_sizes)
        self.pairwise_head = nn.Sequential(
            nn.Linear(512, 256),
            GeGLU(),
            nn.Dropout(pairwise_dropout),
            nn.Linear(128, _PAIRWISE_DIM),
        )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(_REL_FEATURE_DIM + _PAIRWISE_DIM),
            nn.Linear(_REL_FEATURE_DIM + _PAIRWISE_DIM, 512),
            GeGLU(),
            nn.Linear(256, highway_dim)
        )
        self.blocks = nn.ModuleList()
        for i, target_dim in enumerate(self.downsample_schedule):
            self.blocks.append(PurelyRelationalBlock(highway_dim, target_dim))
            if self.mlp_sizes[i] > 0:
                self.blocks.append(MLPBlock(highway_dim, target_dim))
        self.output_head = nn.Sequential(
            nn.Linear(highway_dim, 128),
            GeGLU(),
            nn.Linear(64, 1)
        )

    def forward(self, embeddings, transforms, mask):
        """Forward pass.

        Args:
            embeddings: (B, N, 256) - fragment embeddings from backbone
            transforms: (B, N, 16) - fragment transforms (4x4 pose matrices flattened)
            mask: (B, N) - 1 for real fragments, 0 for padding

        Returns:
            affinity_logits: (B, N, N) - pairwise same-asset affinity logits
        """
        B, N, _ = transforms.shape
        translation, scale, rot = transforms_to_pose(transforms, mask)
        rel_feats = compute_relative_features(
            translation, scale.view(B, N),
            translation, scale.squeeze(-1),
            rot, rot
        )
        # Pairwise from embeddings: |e_i - e_j| and e_i * e_j
        emb_i = embeddings.unsqueeze(1)  # (B, 1, N, 256)
        emb_j = embeddings.unsqueeze(2)  # (B, N, 1, 256)
        abs_diff = torch.abs(emb_i - emb_j)  # (B, N, N, 256)
        prod = emb_i * emb_j  # (B, N, N, 256)
        pairwise_feats = torch.cat([abs_diff, prod], dim=-1)  # (B, N, N, 512)
        pairwise_feats = self.pairwise_head(pairwise_feats)  # (B, N, N, 4)
        # Concat relative features + pairwise features
        combined = torch.cat([rel_feats, pairwise_feats], dim=-1)  # (B, N, N, 26)
        e = self.input_proj(combined)
        for block in self.blocks:
            e = block(e, mask)
        affinity_logits = self.output_head(e).squeeze(-1)
        mask = torch.where(mask < 0.0, torch.zeros_like(mask),
                          torch.where(mask > 1.0, torch.ones_like(mask), mask))
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        affinity_logits_for_loss = affinity_logits + (1.0 - mask_2d) * -1e4
        return affinity_logits_for_loss
