"""Peeler - Purely relational architecture for fragment clustering.

Architecture:
    PurelyRelationalBlock: bottleneck triangular update with fixed highway
    PurelyRelationalPeeler: affinity logits output for BCE training

Training: BCE loss on same-asset affinity matrix.
Optimization: End-to-end training of relational blocks.

Registered with openpoints MODELS registry.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from openpoints.models.build import MODELS


class StableRMSNorm(nn.Module):
    """Stable RMSNorm with clean variance + eps for ONNX and compiling.

    Matches PyTorch's native nn.RMSNorm behavior (scaling-only, no bias) 
    but remains fully exportable on older ONNX Opsets (like Opset 15/17) 
    and highly optimized for compile.
    """
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
        # 1. Compute variance (mean of squares) along the last dimension.
        # No subtraction means zero risk of subtractive cancellation (no NaNs).
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        
        # 2. Scale via rsqrt (which compiles into a single fused CUDA kernel)
        x_normed = x * torch.rsqrt(variance + self.eps)
        
        # 3. Apply the learnable scale (gamma)
        if self.elementwise_affine:
            x_normed = x_normed * self.weight
                
        return x_normed



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
    scale = scale + 1e-8
    rot = mat[:, :, :3, :3].reshape(B, N, -1) / scale

    mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)

    active_translation = translation * mask_expanded

    return active_translation, scale, rot



def compute_relative_features(seed_T, seed_S, cand_T, cand_S, seed_rot, cand_rot, mask):
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

    diff = cand_T.unsqueeze(1) - seed_T.unsqueeze(2)  # (B, S, N, 3)
    dist_raw = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-8) # (B, S, N, 1) Dont use norm for numerical stability
    
    seed_S_exp = seed_S.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
    cand_S_exp = cand_S.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)

    # 1. Reshape 9D flat vectors back to 3x3 rotation matrices
    R_seed = seed_rot.reshape(B, S, 3, 3)
    R_cand = cand_rot.reshape(B, N, 3, 3)

    # 2. Unsqueeze and transpose seed rotation for batched matrix multiplication
    R_seed_T = R_seed.unsqueeze(2).transpose(-2, -1)  # (B, S, 1, 3, 3)
    R_cand_exp = R_cand.unsqueeze(1)                  # (B, 1, N, 3, 3)

    # 3. Compute relative rotation: [B, S, N, 3, 3]
    R_rel = torch.matmul(R_seed_T, R_cand_exp)

    # 4. Flatten the 3x3 relative rotation matrix back into a 9D vector
    # This features is now 100% rotation-invariant!
    rot_diff = R_rel.reshape(B, S, N, 9)

    # 5. Optimized Cosine Similarity using the Trace
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2] # (B, S, N)
    rot_cosine = (trace / 3.0).unsqueeze(-1)                      # (B, S, N, 1)

    direction_local = torch.matmul(R_seed_T, (diff / (dist_raw.relu() + 1e-8)).unsqueeze(-1)).squeeze(-1) # (B, S, N, 3)

    # NEW FEATURE 1: Neighborhood-Relative Distance with standard log10 dynamic range scaling
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
    num_active = mask.sum(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)  # (B, 1, 1, 1)
    num_active_safe = torch.clamp(num_active, min=1.0) # Defensively protect divisor
    
    sum_dist = torch.sum(dist_raw * mask_expanded, dim=2, keepdim=True)  # (B, S, 1, 1)
    mean_dist = sum_dist / num_active_safe  # (B, S, 1, 1)
    
    rel_dist_ratio = dist_raw / (mean_dist + 1e-8)
    rel_dist_to_mean = (torch.log10(rel_dist_ratio + 1e-8) / 8) * mask_expanded  # (B, S, N, 1)

    # NEW FEATURE 2: Physical-Size Relative Distance with standard log10 dynamic range scaling
    sum_scale = torch.sum(cand_S_exp * mask_expanded, dim=2, keepdim=True)  # (B, 1, 1, 1)
    mean_scale = sum_scale / num_active_safe  # (B, 1, 1, 1)
    
    scale_dist_ratio = dist_raw / (mean_scale + 1e-8)
    dist_to_mean_scale = (torch.log10(scale_dist_ratio + 1e-8) / 8) * mask_expanded  # (B, S, N, 1)

    # FEATURE 3: Competitive Proximity Attention (Softmax with temperature 0.5)
    # Masked softmax: set masked-out elements to a very large negative value before softmax
    # so they evaluate to exactly 0.0 attention score.
    masked_rel_dist = rel_dist_ratio + (1.0 - mask_expanded) * 1e4
    proximity_attention = torch.softmax(-masked_rel_dist / 0.5, dim=2)  # (B, S, N, 1)

    return torch.cat([
        direction_local,  # direction (3) - safe division
        (torch.log10(((dist_raw - seed_S_exp - cand_S_exp).relu() + 1e-8) / (seed_S_exp + 1e-8)) / 8),  # dist_bwn_normalized_s (1)
        (torch.log10(dist_raw / (seed_S_exp + 1e-8) + 1e-8) / 8),                        # dist_normalized_s (1)
        (torch.log10(cand_S_exp / (seed_S_exp + 1e-8) + 1e-8) / 8),                             # rel_scale (1)
        rot_diff,                                                               # rot_diff (9)
        rot_cosine,                                                             # rot_cosine (1)
        rel_dist_to_mean,                                                                # Neighborhood relative distance (1)
        dist_to_mean_scale,                                                              # Physical-size relative distance (1)
        proximity_attention
    ], dim=-1)

_REL_FEATURE_DIM = 19

class SwiGLU(nn.Module):
    """
    ONNX-safe Gated Linear Unit with GELU activation.
    Splits the last dimension in half and uses one half to gate the other.
    """
    def forward(self, x):
        # chunk(2, dim=-1) splits the channel dimension into two equal tensors
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.nn.functional.silu(x2)
    
class RelationalAttentionBlock(nn.Module):
    """Competitive Relational Edge Attention Block."""
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
        
        # PROJECTION DROPOUT (Applied to the branch output)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.res_scale = nn.Parameter(0.1 * torch.ones(highway_dim))

    def forward(self, e, mask):
        B, N, _, D = e.shape
        norm = self.norm(e)  # (B, N, N, D)
        
        x = norm.view(B * N, N, D)
        
        q = self.q_proj(x).view(B * N, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B * N, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B * N, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        mask_bool = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mask_bool = mask_bool.expand(B, N, self.num_heads, N, N)
        mask_bool = mask_bool.reshape(B * N, self.num_heads, N, N).to(torch.bool)
        
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_bool,
            dropout_p=self.attn_dropout if self.training else 0.0
        )
        
        out = out.transpose(1, 2).reshape(B, N, N, -1)
        
        # Project up, apply dropout to the branch update
        result = self.out_proj(out)
        result = self.proj_dropout(result)  # <--- Drop out here
        
        mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        result = result * mask_2d
        
        # The highway "e" remains untouched by dropout
        return e + self.res_scale * result
    
class PurelyRelationalBlock(nn.Module):
    """Bottleneck Purely Relational Block with fixed highway."""
    def __init__(self, highway_dim, target_dim, dropout=0.0):
        super().__init__()
        self.norm = StableRMSNorm(highway_dim)
        
        self.left_proj = nn.Sequential(
            nn.Linear(highway_dim, target_dim),
            nn.GELU()
        )
        self.right_proj = nn.Sequential(
            nn.Linear(highway_dim, target_dim),
            nn.GELU()
        )
        self.proj_up = nn.Sequential(
            nn.Linear(target_dim, highway_dim * 2),
            SwiGLU(),
            nn.Linear(highway_dim, highway_dim)
        )

        self.res_scale = nn.Parameter(0.1 * torch.ones(highway_dim))
        self.dropout = nn.Dropout(dropout)  # Regulates the final branch output

    def forward(self, e, mask):
        norm = self.norm(e)
        left = self.left_proj(norm)
        right = self.right_proj(norm)

        mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        left = left * mask_2d
        right = right * mask_2d

        left_perm = left.permute(0, 3, 1, 2)
        right_perm = right.permute(0, 3, 1, 2)

        num_active = mask.sum(dim=-1, keepdim=True)
        scale_factor = torch.sqrt(num_active).unsqueeze(-1).unsqueeze(-1)

        triangular_sum = torch.matmul(left_perm, right_perm) / scale_factor
        triangular_out = triangular_sum.permute(0, 2, 3, 1)

        # Project up, then apply dropout to the branch update
        result = self.proj_up(triangular_out)
        result = self.dropout(result)  # <--- Drop out here
        result = result * mask_2d

        return e + self.res_scale * result


class MLPBlock(nn.Module):
    """MLP residual block on pairwise feature tensor."""
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

    def forward(self, e, mask):
        result = self.mlp(e)
        result = self.dropout(result)  # <--- Drop out the update branch
        
        mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        result = result * mask_2d
        return e + self.res_scale * result

class Symmetrizer(nn.Module):
    def __init__(self, input, target):
        super().__init__()

        self.scaler = nn.Linear(input, target)

    def forward(self, e):
        e = self.scaler(e)

        e_T = e.transpose(1, 2)

        h_sum = (e + e_T)                     # Mutual confidence
        h_diff = torch.abs(e - e_T)         # Conflict/Disagreement
        h_prod = e * e_T                    # Intersection/Consensus
        
        h_sym = torch.cat([h_sum, h_diff, h_prod], dim=-1)  # (B, N, N, 3 * H)
        
        return h_sym
        


_PAIRWISE_DIM = 32



@MODELS.register_module()
class PurelyRelationalPeeler(nn.Module):
    def __init__(self, downsample_schedule, mlp_sizes, highway_dim, pairwise_dropout, attn_dropout=0.0, output_dropout=0.0, **kwargs):
        super().__init__()
        self.downsample_schedule = list(downsample_schedule)
        self.num_blocks = len(self.downsample_schedule)
        self.mlp_sizes = list(mlp_sizes)
        
        self.pairwise_head = nn.Sequential(
            nn.Linear(512, 512),
            SwiGLU(),
            nn.Linear(256, _PAIRWISE_DIM),
        )

        _HIDDEN_REL = 32
        _HIDDEN_FEATS = _HIDDEN_REL + _PAIRWISE_DIM

        self.rel_head = nn.Sequential(
            nn.Linear(_REL_FEATURE_DIM, 64),
            SwiGLU(),
            nn.Linear(32, _HIDDEN_REL)
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
                    highway_dim, 
                    target_dim, 
                    attn_dropout=attn_dropout, 
                    proj_dropout=pairwise_dropout  # Pass pairwise dropout to projection
                ))
            else:
                self.blocks.append(PurelyRelationalBlock(
                    highway_dim, 
                    target_dim, 
                    dropout=pairwise_dropout       # Pass pairwise dropout to projection
                ))

            if self.mlp_sizes[i] > 0:
                self.blocks.append(MLPBlock(
                    highway_dim, 
                    target_dim, 
                    dropout=pairwise_dropout       # Pass pairwise dropout to MLP
                ))

        self.output_head = nn.Sequential(
            StableRMSNorm(highway_dim),
            nn.Dropout(output_dropout),
            Symmetrizer(highway_dim, highway_dim),
            nn.Linear(highway_dim * 3, highway_dim),
            nn.GELU(),
            nn.Linear(highway_dim, 1)
        )
        # REMOVED: self.block_dropout. Block regularizations are now internal.

    def forward(self, embeddings, transforms, mask):
        B, N, _ = transforms.shape
        translation, scale, rot = transforms_to_pose(transforms, mask)
        rel_feats = compute_relative_features(
            translation, scale.squeeze(-1),
            translation, scale.squeeze(-1),
            rot, rot, mask
        )

        norm_embeddings = embeddings / 6
        norm_embeddings = torch.clamp(norm_embeddings, max=2, min=-2)

        # Pairwise from embeddings: |e_i - e_j| and e_i * e_j
        emb_i = norm_embeddings.unsqueeze(1)  # (B, 1, N, 256)
        emb_j = norm_embeddings.unsqueeze(2)  # (B, N, 1, 256)

        abs_diff = torch.abs(emb_i - emb_j)  # (B, N, N, 256)

        prod = emb_i * emb_j  # (B, N, N, 256)

        pairwise_feats = torch.cat([abs_diff, prod], dim=-1)  # (B, N, N, 512)
        pairwise_feats = self.pairwise_head(pairwise_feats)  # (B, N, N, 4)
        rel_feats = self.rel_head(rel_feats)
        
        # Concat relative features + pairwise features
        combined = torch.cat([rel_feats, pairwise_feats], dim=-1)  # (B, N, N, 80)
        e = self.input_proj(combined)
        
        for block in self.blocks:
            e = block(e, mask)
        affinity_logits = self.output_head(e).squeeze(-1)
        mask = torch.where(mask < 0.0, torch.zeros_like(mask),
                          torch.where(mask > 1.0, torch.ones_like(mask), mask))
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        affinity_logits_for_loss = affinity_logits + (1.0 - mask_2d) * -1e4
        return affinity_logits_for_loss