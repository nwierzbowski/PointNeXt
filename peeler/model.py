"""Peeler - Purely relational sparse architecture with Top-K competition.

Architecture:
    PurelyRelationalBlock: bottleneck triangular update with fixed highway (N x K sparse)
    RelationalAttentionBlock: sparse local attention (N x K sparse)
    PurelyRelationalPeeler: sparse Top-K competition logits for softmax CE training

Training: Cross-entropy over K candidates per node. Multi-scene batched via scene_ids.
Inference: Single scene (N, 256) x (N, 16) -> (N, K) logits + (N, K) indices. ONNX exportable.

Registered with openpoints MODELS registry.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from openpoints.models.build import MODELS


class StableRMSNorm(nn.Module):
    """Stable RMSNorm with clean variance + eps for ONNX and compiling."""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
            self.num_features = normalized_shape
        else:
            self.normalized_shape = tuple(normalized_shape)
            self.num_features = normalized_shape[-1]

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
        else:
            self.register_parameter('rmsnorm_weight', None)

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        if self.elementwise_affine:
            x_normed = x_normed * self.weight
        return x_normed


def transforms_to_pose(transforms):
    """Extract translation, scale, and normalized rotation from 4x4 transforms."""
    N = transforms.shape[0]
    mat = transforms.view(N, 4, 4)
    translation = mat[:, :3, 3]
    scale = torch.norm(mat[:, :3, :3], dim=-1).mean(-1, keepdim=True)
    scale = scale + 1e-8
    rot = mat[:, :3, :3].reshape(N, -1) / scale

    return translation, scale, rot


def compute_relative_features(T, S, R, topk_indices):
    """Compute sparse relative features between seed and top-K candidate poses."""
    N = T.shape[0]
    K = topk_indices.shape[-1]

    seed_T_flat = T.unsqueeze(1)
    seed_S_flat = S.unsqueeze(1)
    seed_rot_flat = R.unsqueeze(1)

    cand_T_flat = T[topk_indices, :]
    cand_S_flat = S[topk_indices]
    cand_rot_flat = R[topk_indices, :]

    diff = cand_T_flat - seed_T_flat
    dist_raw = torch.sqrt(diff.square().sum(-1, keepdim=True) + 1e-8)

    seed_S_exp = seed_S_flat.unsqueeze(-1)
    cand_S_exp = cand_S_flat.unsqueeze(-1)

    R_seed = seed_rot_flat.reshape(N, 1, 3, 3)
    R_cand = cand_rot_flat.reshape(N, K, 3, 3)
    R_seed_T = R_seed.transpose(-2, -1)
    R_rel = torch.matmul(R_seed_T, R_cand)
    rot_diff = R_rel.reshape(N, K, 9)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    rot_cosine = (trace / 3.0).unsqueeze(-1)
    direction_local = torch.matmul(R_seed_T, (diff / (dist_raw.relu() + 1e-8)).unsqueeze(-1)).squeeze(-1)

    mean_dist = dist_raw.mean(dim=1, keepdim=True)
    rel_dist_ratio = dist_raw / (mean_dist + 1e-8)
    rel_dist_to_mean = torch.log10(rel_dist_ratio + 1e-8) / 8

    mean_scale = cand_S_exp.mean(dim=1, keepdim=True)
    scale_dist_ratio = dist_raw / (mean_scale + 1e-8)
    dist_to_mean_scale = torch.log10(scale_dist_ratio + 1e-8) / 8

    return torch.cat([
        direction_local,
        (torch.log10(((dist_raw - seed_S_exp - cand_S_exp).relu() + 1e-8) / (seed_S_exp + 1e-8)) / 8),
        (torch.log10(dist_raw / (seed_S_exp + 1e-8) + 1e-8) / 8),
        (torch.log10(cand_S_exp / (seed_S_exp + 1e-8) + 1e-8) / 8),
        rot_diff,
        rot_cosine,
        rel_dist_to_mean,
        dist_to_mean_scale,
    ], dim=-1)


_REL_FEATURE_DIM = 18


class SwiGLU(nn.Module):
    """ONNX-safe Gated Linear Unit with SiLU activation."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.nn.functional.silu(x2)


class RelationalAttentionBlock(nn.Module):
    """Competitive Relational Edge Attention Block (N x K sparse)."""
    def __init__(self, highway_dim, target_dim, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm = StableRMSNorm(highway_dim)
        self.num_heads = num_heads
        self.target_dim = target_dim
        self.head_dim = target_dim // num_heads
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(highway_dim, target_dim)
        self.k_proj = nn.Linear(highway_dim, target_dim)
        self.v_proj = nn.Linear(highway_dim, target_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(target_dim, highway_dim * 2),
            SwiGLU(),
            nn.Linear(highway_dim, highway_dim)
        )

        self.proj_dropout = nn.Dropout(proj_dropout)
        self.res_scale = nn.Parameter(0.1 * torch.ones(highway_dim))

    def forward(self, e, candidate_mask):
        N, K, D = e.shape
        norm = self.norm(e)

        q = self.q_proj(norm).view(N, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm).view(N, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm).view(N, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = candidate_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)
        all_zero = (~attn_mask).all(dim=-1, keepdim=True)
        attn_mask = attn_mask | all_zero

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).reshape(N, K, -1)
        result = self.out_proj(out)
        result = self.proj_dropout(result)
        result = result * candidate_mask.unsqueeze(-1)

        return e + self.res_scale * result


class PurelyRelationalBlock(nn.Module):
    """Bottleneck Sparse Triangular Relational Block (AlphaFold 2 2-Hop Path Update).
    
    Projects highway_dim -> target_dim bottleneck, aggregates real 2-hop edge paths 
    (i -> m_v -> m_u) via sparse topk_indices einsum contraction, and projects up.
    """
    def __init__(self, highway_dim, target_dim, dropout=0.0):
        super().__init__()
        self.norm = StableRMSNorm(highway_dim)

        self.left_proj = nn.Linear(highway_dim, target_dim)
        self.right_proj = nn.Linear(highway_dim, target_dim)

        self.proj_up = nn.Sequential(
            nn.Linear(target_dim, highway_dim * 2),
            SwiGLU(),
            nn.Linear(highway_dim, highway_dim)
        )

        self.res_scale = nn.Parameter(0.1 * torch.ones(highway_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, e, candidate_mask, topk_indices):
        N, K, D = e.shape
        c_mask = candidate_mask.unsqueeze(-1)
        norm = self.norm(e)

        a = self.left_proj(norm) * c_mask   # (N, K, target_dim)
        b = self.right_proj(norm) * c_mask # (N, K, target_dim)

        b_intermediate = b[topk_indices]  # Advanced indexing: (N, K, K, target_dim)
        
        triangular_out = torch.einsum("nvd, nvud -> nud", a, b_intermediate) / math.sqrt(K)

        result = self.proj_up(triangular_out)
        result = self.dropout(result) * c_mask

        return e + self.res_scale * result


class MLPBlock(nn.Module):
    """MLP residual block on pairwise feature tensor (N x K sparse)."""
    def __init__(self, highway_dim, target_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            StableRMSNorm(highway_dim),
            nn.Linear(highway_dim, target_dim * 2),
            SwiGLU(),
            nn.Linear(target_dim, highway_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(0.1 * torch.ones(highway_dim))

    def forward(self, e, candidate_mask):
        result = self.mlp(e)
        result = self.dropout(result)
        result = result * candidate_mask.unsqueeze(-1)
        return e + self.res_scale * result


_PAIRWISE_DIM = 128


@MODELS.register_module()
class PurelyRelationalPeeler(nn.Module):
    def __init__(self, downsample_schedule, mlp_sizes, highway_dim, pairwise_dropout, attn_dropout=0.0, output_dropout=0.0, top_k=32, temperature=0.1, **kwargs):
        super().__init__()
        self.downsample_schedule = list(downsample_schedule)
        self.num_blocks = len(self.downsample_schedule)
        self.mlp_sizes = list(mlp_sizes)
        self.top_k = top_k
        self.temperature = temperature

        self.pairwise_head = nn.Sequential(
            nn.Linear(512, 512),
            SwiGLU(),
            nn.Linear(256, _PAIRWISE_DIM),
        )

        _HIDDEN_REL = 64
        _HIDDEN_FEATS = _HIDDEN_REL + _PAIRWISE_DIM

        self.rel_head = nn.Sequential(
            nn.Linear(_REL_FEATURE_DIM, 128),
            SwiGLU(),
            nn.Linear(64, _HIDDEN_REL)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(_HIDDEN_FEATS, _HIDDEN_FEATS * 2),
            SwiGLU(),
            StableRMSNorm(_HIDDEN_FEATS),
            nn.Linear(_HIDDEN_FEATS, highway_dim),
            nn.Dropout(pairwise_dropout),
        )

        self.blocks = nn.ModuleList()
        for i, target_dim in enumerate(self.downsample_schedule):
            if i % 2 == 0:
                self.blocks.append(RelationalAttentionBlock(
                    highway_dim, target_dim,
                    attn_dropout=attn_dropout, proj_dropout=pairwise_dropout
                ))
            else:
                self.blocks.append(PurelyRelationalBlock(
                    highway_dim, target_dim, dropout=pairwise_dropout
                ))
            if self.mlp_sizes[i] > 0:
                self.blocks.append(MLPBlock(
                    highway_dim, target_dim, dropout=pairwise_dropout
                ))

        self.output_head = nn.Sequential(
            StableRMSNorm(highway_dim),
            nn.Dropout(output_dropout),
            nn.Linear(highway_dim, highway_dim),
            nn.GELU(),
            nn.Linear(highway_dim, 1)
        )

    def forward(self, embeddings, transforms, scene_ids=None):
        """Forward pass.

        Single scene or multi-scene: scene_ids optional
            Input: (N_total, 256), (N_total, 16), [(N_total,)] scene_ids
            Output: (N_total, K) logits, (N_total, K) topk_indices, (N_total, K) candidate_mask
        """
        N_total = embeddings.shape[0]
        K = self.top_k

        translation, scale, rot = transforms_to_pose(transforms)
        
        if scene_ids is None:
            topk_indices, candidate_mask = self._get_topk_neighbors_single(translation, K)
        else:
            topk_indices, candidate_mask = self._get_topk_neighbors_scene(
                translation, scene_ids, K
            )

        rel_feats = compute_relative_features(
            translation, scale.squeeze(-1),
            rot, topk_indices
        )

        norm_emb = torch.clamp(embeddings / 6, max=2, min=-2)
        emb_neighbors = norm_emb[topk_indices, :]
        emb_self = norm_emb.unsqueeze(1)

        pairwise_feats = self.pairwise_head(
            torch.cat([torch.abs(emb_self - emb_neighbors), emb_self * emb_neighbors], dim=-1)
        )
        rel_feats = self.rel_head(rel_feats)

        e = self.input_proj(torch.cat([rel_feats, pairwise_feats], dim=-1))  # (N_total, K, D)

        for block in self.blocks:
            if isinstance(block, PurelyRelationalBlock):
                e = block(e, candidate_mask, topk_indices=topk_indices)
            else:
                e = block(e, candidate_mask)

        logits = self.output_head(e).squeeze(-1)  # (N_total, K)
        return logits, topk_indices, candidate_mask

    def _get_topk_neighbors_single(self, translation, k):
        """Single-scene KNN: fully vectorized, ONNX-compatible (no loops).

        Excludes self-relations. Pads to K via modulo indexing when N < K.
        """
        N_total = translation.shape[0]
        device = translation.device

        # Compute pairwise distance matrix (N, N)
        diff = translation.unsqueeze(0) - translation.unsqueeze(1)  # (N, N, 3)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # (N, N)

        # Exclude self-relations (dist=0 provides no useful signal)
        row_idx = torch.arange(N_total, device=device).unsqueeze(1)
        col_idx = torch.arange(N_total, device=device).unsqueeze(0)
        dist = dist.masked_fill(row_idx == col_idx, float('inf'))

        # Sort distances ascending (N, N)
        sorted_indices = torch.argsort(dist, dim=-1)  # (N, N)

        # Dynamic Modulo Padding (handles N < K for any N >= 1)
        k_range = torch.arange(k, device=device)        # (K,)
        pad_indices = k_range % N_total                 # (K,) -> maps [0..K-1] back into [0..N-1]
        topk_indices = sorted_indices[:, pad_indices]   # (N, K)

        # Candidate mask: valid = first min(N-1, K) columns (self excluded)
        candidate_mask = (k_range.unsqueeze(0) < (N_total - 1)).expand(N_total, k)  # (N, K)

        return topk_indices, candidate_mask

    def _get_topk_neighbors_scene(self, translation, scene_ids, k):
        """Multi-scene KNN: delegates to _get_topk_neighbors_single per scene."""
        N_total = translation.shape[0]
        device = translation.device
        topk_indices = torch.zeros((N_total, k), dtype=torch.long, device=device)
        candidate_mask = torch.zeros((N_total, k), dtype=torch.bool, device=device)

        unique_scenes = torch.unique(scene_ids)

        for s in unique_scenes:
            idx = (scene_ids == s).nonzero(as_tuple=True)[0]
            t_s = translation[idx]

            # Delegate to single-scene function (keeps both paths aligned)
            local_topk, mask_s = self._get_topk_neighbors_single(t_s, k)

            # Map local indices back to global indices
            topk_indices[idx] = idx[local_topk]
            candidate_mask[idx] = mask_s

        return topk_indices, candidate_mask


