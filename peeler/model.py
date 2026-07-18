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

    direction_local = torch.matmul(R_seed_T, (diff / (dist_raw.relu() + 1e-8)).unsqueeze(-1)).squeeze(-1) # (B, S, N, 3)

    # print("max: ", torch.log10(torch.clamp(torch.clamp(dist_raw - seed_S_exp - cand_S_exp, min=1e-8), min=1e-8)) / 8 + 1)
    # print("min: ", torch.log10(torch.clamp(torch.clamp(dist_raw - seed_S_exp - cand_S_exp, min=1e-8), min=1e-8)) / 8 + 1)

    return torch.cat([
        direction_local,  # direction (3) - safe division
        (torch.log10(((dist_raw - seed_S_exp - cand_S_exp).relu() + 1e-8) / (seed_S_exp + 1e-8)) / 8),  # dist_bwn_normalized_s (1)
        (torch.log10(dist_raw / (seed_S_exp + 1e-8) + 1e-8) / 8),                        # dist_normalized_s (1)
        (torch.log10(cand_S_exp / (seed_S_exp + 1e-8) + 1e-8) / 8),                             # rel_scale (1)
        rot_diff,                                                               # rot_diff (9)
        rot_cosine,                                                             # rot_cosine (1)
    ], dim=-1)

_REL_FEATURE_DIM = 16

class SwiGLU(nn.Module):
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

    def forward(self, e, mask):
        norm = self.norm(e)
        left = self.left_proj(norm)
        right = self.right_proj(norm)

        mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        left = left * mask_2d
        right = right * mask_2d

        left_perm = left.permute(0, 3, 1, 2)
        right_perm = right.permute(0, 3, 1, 2)

        num_active = mask.sum(dim=-1, keepdim=True) # (B, 1)
        scale_factor = torch.sqrt(num_active).unsqueeze(-1).unsqueeze(-1) # (B, 1, 1, 1)

        triangular_sum = torch.matmul(left_perm, right_perm) / scale_factor
        triangular_out = triangular_sum.permute(0, 2, 3, 1)

        result = self.proj_up(triangular_out)
        result = result * mask_2d

        # self.res_scale = nn.Parameter(torch.tensor(0.1))

        return e + result


class MLPBlock(nn.Module):
    """MLP residual block on pairwise feature tensor.

    Projects down to target_dim, applies MLP, projects back up to highway_dim.
    Element-wise operations on (B, N, N, D) with bottleneck architecture.
    """

    def __init__(self, highway_dim, target_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            StableRMSNorm(highway_dim),
            nn.Linear(highway_dim, target_dim * 2),
            SwiGLU(),
            nn.Linear(target_dim, highway_dim)
        )

    def forward(self, e, mask):
        result = self.mlp(e)
        mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        result = result * mask_2d
        return e + result

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
        


_PAIRWISE_DIM = 48

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
            nn.Linear(512, 512),
            SwiGLU(),
            nn.Linear(256, _PAIRWISE_DIM),
        )

        self.rel_head = nn.Sequential(
            nn.Linear(_REL_FEATURE_DIM, 64),
            SwiGLU(),
            nn.Linear(32, _REL_FEATURE_DIM * 2)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(_REL_FEATURE_DIM * 2 + _PAIRWISE_DIM, (_REL_FEATURE_DIM* 2 + _PAIRWISE_DIM) * 2),
            SwiGLU(),
            StableRMSNorm(_REL_FEATURE_DIM * 2 + _PAIRWISE_DIM),
            nn.Linear(_REL_FEATURE_DIM * 2 + _PAIRWISE_DIM, highway_dim),
        )
        self.blocks = nn.ModuleList()
        for i, target_dim in enumerate(self.downsample_schedule):
            self.blocks.append(PurelyRelationalBlock(highway_dim, target_dim))
            if self.mlp_sizes[i] > 0:
                self.blocks.append(MLPBlock(highway_dim, target_dim))

        self.output_head = nn.Sequential(
            StableRMSNorm(highway_dim),
            Symmetrizer(highway_dim, highway_dim),
            nn.Linear(highway_dim * 3, highway_dim),
            nn.GELU(),
            nn.Linear(highway_dim, 1)
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
            translation, scale.squeeze(-1),
            translation, scale.squeeze(-1),
            rot, rot
        )

        # print("rel_feats: ", torch.max(rel_feats), torch.min(rel_feats))

        # norm_embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        norm_embeddings = embeddings / 6
        norm_embeddings = torch.clamp(norm_embeddings, max=2, min=-2)
        # print("norm_embeddings: ", torch.max(norm_embeddings), torch.min(norm_embeddings))

        # Pairwise from embeddings: |e_i - e_j| and e_i * e_j
        emb_i = norm_embeddings.unsqueeze(1)  # (B, 1, N, 256)
        emb_j = norm_embeddings.unsqueeze(2)  # (B, N, 1, 256)

        abs_diff = torch.abs(emb_i - emb_j)  # (B, N, N, 256)
        # print("abs_diff: ", torch.max(abs_diff), torch.min(abs_diff))

        prod = emb_i * emb_j  # (B, N, N, 256)
        # print("prod: ", torch.max(prod), torch.min(prod))

        pairwise_feats = torch.cat([abs_diff, emb_i - emb_j], dim=-1)  # (B, N, N, 512)
        pairwise_feats = self.pairwise_head(pairwise_feats)  # (B, N, N, 4)
        rel_feats = self.rel_head(rel_feats)
        
        # Concat relative features + pairwise features
        combined = torch.cat([rel_feats, pairwise_feats], dim=-1)  # (B, N, N, 26)
        e = self.input_proj(combined)

        # print("e: ", torch.max(e), torch.min(e))
        
        for block in self.blocks:
            e = block(e, mask)
        affinity_logits = self.output_head(e).squeeze(-1)
        mask = torch.where(mask < 0.0, torch.zeros_like(mask),
                          torch.where(mask > 1.0, torch.ones_like(mask), mask))
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        affinity_logits_for_loss = affinity_logits + (1.0 - mask_2d) * -1e4
        return affinity_logits_for_loss
