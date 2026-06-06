"""Adaptive Object Peeler - Two-model ONNX architecture.

Architecture:
    Model 1 - PeelerBackbone: MLP projection → self-attention → max pool
        Input: transforms(N, 16)
        Output: scene_vec(1, 16)
    
    Model 2 - PeelerLoop: anchor scoring + membership logits
        Input: scene_vec(1,16), transforms(N,16), embeddings(N,256), mask(N)
        Output: anchor_score, membership_logits(N)
    
    Model 3 - Peeler (joint training): combines both models
        - PeelerBackbone runs once to get scene_vec
        - PeelerLoop runs iteratively (N times) for per-fragment scores

Training: full NxN membership matrix, expected loss weighted by P_anchor.
Joint optimization: both backbone and heads receive gradients.

ONNX Export: Two separate models for clean export without dynamic shapes.

Registered with openpoints MODELS registry.
"""
import torch
import torch.nn as nn

from openpoints.models.build import MODELS

# Feature dimension for backbone (16D features all the way through)
_FEAT_DIM = 16

# Raw relative feature dimension: dist(1) + direction(3) = 4
_REL_FEATURE_DIM = 4


class SimpleAttentionBlock(nn.Module):
    """ONNX-compatible transformer block with self-attention and MLP."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PeelerBackbone(nn.Module):
    """MLP projection → self-attention → max pool backbone.

    Input: transforms(N, 16)
    Output: scene_vec(1, 16)
    """

    def __init__(
        self,
        model_dim,  # pass-through, unused
        attention_heads=4,
        attention_blocks=4,
    ):
        super().__init__()

        self.proj = nn.Linear(4, _FEAT_DIM)
        self.blocks = nn.ModuleList([
            SimpleAttentionBlock(_FEAT_DIM, attention_heads)
            for _ in range(attention_blocks)
        ])
        self.norm = nn.LayerNorm(_FEAT_DIM)

    def forward(self, transforms):
        """Forward pass for backbone.

        Args:
            transforms: (B, N, 16) - fragment transforms

        Returns:
            scene_vec: (B, 1, 16) - global scene vector
        """
        B, N, _ = transforms.shape
        mat = transforms.view(B, N, 4, 4)

        translation = mat[:, :, :3, 3]
        scale = torch.norm(mat[:, :, :3, :3], dim=-1).mean(-1, keepdim=True)
        scale = torch.clamp(scale, min=1e-8)

        pos = translation  # (B, N, 3)
        x = torch.cat([pos, scale], dim=-1)  # (B, N, 4)
        x = self.proj(x)  # (B, N, 16)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global max pool over N
        scene_vec = x.max(dim=1, keepdim=True)[0]  # (B, 1, 16)

        return scene_vec


class PeelerLoop(nn.Module):
    """Anchor scoring + membership logits with ONNX-optimized export path.

    Training: full NxN membership matrix for gradient flow to all anchors.
    ONNX Export: only computes 1×N for best anchor (avoids NxN computation).

    Input: scene_vec(1,16), transforms(N,16), embeddings(N,256), mask(N)
    Output: anchor_scores(N), membership_logits(N) [export] or (N,N) [training]
    """

    def __init__(self, model_dim, anchor_drop_rate, relation_drop_rate):
        super().__init__()
        self.model_dim = model_dim  # pass-through, unused

        # Anchor head: MLP that scores each fragment as a potential anchor seed
        # Uses pose features (translation + scale) instead of embeddings
        # High scores -> complex, identifiable parts (receiver, barrel)
        # Low scores -> simple, redundant parts (screws, noise)
        self.anchor_pose_proj = nn.Linear(4, _FEAT_DIM)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(_FEAT_DIM * 2, _FEAT_DIM),
            nn.GELU(),
            nn.Dropout(anchor_drop_rate),
            nn.Linear(_FEAT_DIM, 1),
        )

        # Relation head: MLP that computes membership logits from relative features
        self.relation_mlp = nn.Sequential(
            nn.Linear(_REL_FEATURE_DIM, 128),
            nn.GELU(),
            nn.Dropout(relation_drop_rate),
            # nn.Linear(128, 128),
            # nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(relation_drop_rate),
            nn.Linear(64, 1),
        )

    def forward(self, scene_vec, transforms, embeddings=None, mask=None):
        """Forward pass with ONNX-optimized export path.

        Args:
            scene_vec: (B, 1, 16) - scene vector from backbone
            transforms: (B, N, 16) - fragment transforms
            embeddings: (B, N, 256) - all fragment embeddings (not used)
            mask: (B, N) - 1 for real fragments (training only)

        Returns:
            anchor_scores: (B, N) - anchor logits for all fragments
            membership_logits: (B, N) for ONNX export, (B, N, N) for training
        """
        B, N, _ = transforms.shape
        # B, C, _ = seed_pose.shape
        # _, N, _ = cand_pose.shape

        # Extract pose features (translation + scale) from transforms for anchor head
        mat = transforms.view(B, N, 4, 4)
        translation = mat[:, :, :3, 3]
        scale = torch.norm(mat[:, :, :3, :3], dim=-1).mean(-1, keepdim=True)
        scale = torch.clamp(scale, min=1e-8)
        pose_input = torch.cat([translation, scale], dim=-1)  # (B, N, 4)

        # Anchor head: score ALL fragments as anchors in parallel
        pose_proj = self.anchor_pose_proj(pose_input)  # (B, N, 16)
        scene_expanded = scene_vec.expand(-1, N, _FEAT_DIM)  # (B, N, 16)
        context = torch.cat([scene_expanded, pose_proj], dim=-1)  # (B, N, 32)
        anchor_scores = self.anchor_mlp(context).squeeze(-1)  # (B, N)

        # Decompose transforms -> full pose features (for relation head)
        pose_features = self._decompose_pose(transforms)  # (B, N, 13)

        if torch.onnx.is_in_onnx_export():
            # ONNX export: only compute affinities for top anchor (avoids NxN)
            seed_idx = torch.argmax(anchor_scores, dim=1)  # (B,)

            # Gather seed pose
            seed_idx_expanded = seed_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, 13)  # (B, 1, 13)
            seed_pose = torch.gather(pose_features, 1, seed_idx_expanded)  # (B, 1, 13)

            # Compute relative features: seed vs all N candidates → (B, 1, N, 4)
            rel_features = self._compute_relative_features(seed_pose, pose_features)  # (B, 1, N, 4)

            # Relation head → (B, 1, N, 1) → (B, 1, N) → (B, N)
            membership_logits = self.relation_mlp(rel_features).squeeze(-1)  # (B, 1, N, 1) → (B, 1, N)
            membership_logits = membership_logits.squeeze(1)  # (B, 1, N) → (B, N)
            membership_logits = membership_logits + (1 - mask) * -1e9
        else:
            # Training: full NxN for gradient flow to all anchors
            rel_features = self._compute_relative_features(pose_features, pose_features)  # (B, N, N, 4)

            # Relation head → (B, N, N)
            membership_logits = self.relation_mlp(rel_features).squeeze(-1)  # (B, N, N)

            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
            membership_logits = membership_logits + (1 - mask_2d) * -1e9

        return anchor_scores, membership_logits

    def _decompose_pose(self, transforms):
        """Decompose 4x4 world matrices into pose features.

        Args:
            transforms: (B, N, 16) f32 - row-major 4x4 matrices

        Returns:
            pose_features: (B, N, 13) f32 - [T(3), S(1), R(9)]
        """
        B, N, _ = transforms.shape
        mat = transforms.view(B, N, 4, 4)

        translation = mat[:, :, :3, 3]
        scale = torch.norm(mat[:, :, :3, :3], dim=-1).mean(-1, keepdim=True).unsqueeze(-1)
        
        # DEBUG: Check scale before clamp
        # if not torch.isfinite(scale).all():
        #     nan_count = (~torch.isfinite(scale)).sum().item()
        #     print(f'WARNING: scale has {nan_count} non-finite values BEFORE clamp')
        #     finite_mask = torch.isfinite(scale)
        #     if finite_mask.any():
        #         print(f'  scale stats: min={scale[finite_mask].min():.4f}, max={scale[finite_mask].max():.4f}')
        #     print(f'  transforms stats: min={transforms.min():.4f}, max={transforms.max():.4f}')
        
        scale = torch.clamp(scale, min=1e-8)
        rot = mat[:, :, :3, :3] / scale
        rot_flat = rot.reshape(B, N, -1)

        pose_features = torch.cat([translation, scale.view(B, N, 1), rot_flat], dim=-1)
        
        # DEBUG: Check pose_features before nan_to_num
        # if not torch.isfinite(pose_features).all():
        #     nan_count = (~torch.isfinite(pose_features)).sum().item()
        #     print(f'WARNING: pose_features has {nan_count} non-finite values BEFORE nan_to_num')
        
        # result = torch.nan_to_num(pose_features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # DEBUG: Check result after nan_to_num
        # if not torch.isfinite(result).all():
        #     nan_count = (~torch.isfinite(result)).sum().item()
        #     print(f'WARNING: result has {nan_count} non-finite values AFTER nan_to_num (BUG!)')
        
        return pose_features

    def _compute_relative_features(self, seed_pose, cand_pose):
        """Compute relative features between seed and candidate poses.

        Args:
            seed_pose: (B, S, 13) f32 - seed pose features [T(3), S(1), R(9)]
            cand_pose: (B, N, 13) f32 - candidate pose features [T(3), S(1), R(9)]

        Returns:
            rel_features: (B, S, N, 4) f32 - [dist(1), direction(3)] with scale normalization
        """
        seed_T = seed_pose[:, :, :3]  # (B, S, 3)
        seed_S = seed_pose[:, :, 3]  # (B, S) - scale at index 3
        cand_T = cand_pose[:, :, :3]  # (B, N, 3)
        cand_S = cand_pose[:, :, 3]
        
        # DEBUG: Check seed_S
        # if not torch.isfinite(seed_S).all():
        #     nan_count = (~torch.isfinite(seed_S)).sum().item()
        #     print(f'WARNING: seed_S has {nan_count} non-finite values')
        #     finite_mask = torch.isfinite(seed_S)
        #     if finite_mask.any():
        #         print(f'  seed_S stats: min={seed_S[finite_mask].min():.4f}, max={seed_S[finite_mask].max():.4f}')

        diff = cand_T.unsqueeze(1) - seed_T.unsqueeze(2)  # (B, S, N, 3)
        dist_raw = torch.norm(diff, dim=-1, keepdim=True)  # (B, S, N, 1)
        
        # Normalize distance by seed scale (division in linear space)
        seed_S_expanded = seed_S.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
        cand_S_expanded = cand_S.unsqueeze(1).unsqueeze(-1) # (B, 1, N, 1)
        dist_normalized = dist_raw / (seed_S_expanded)
        
        # DEBUG: Check dist_normalized
        # if not torch.isfinite(dist_normalized).all():
        #     nan_count = (~torch.isfinite(dist_normalized)).sum().item()
        #     print(f'WARNING: dist_normalized has {nan_count} non-finite values')
        #     finite_mask = torch.isfinite(seed_S_expanded)
        #     if finite_mask.any():
        #         print(f'  seed_S_expanded stats: min={seed_S_expanded[finite_mask].min():.4f}, max={seed_S_expanded[finite_mask].max():.4f}')
        
        dist = torch.log1p(dist_normalized)
        
        direction_normalized = diff / (dist_raw + 1e-8)  # (B, S, N, 3)

        return torch.cat([dist, direction_normalized], dim=-1)  # (B, S, N, 4)


@MODELS.register_module()
class Peeler(nn.Module):
    """Adaptive Object Peeler model (joint training).

    Full forward pass (softmax all the way through):
        1. PeelerBackbone: PointNeXt → scene vector
        2. PeelerLoop (iterative): for each fragment as anchor:
            - Anchor scoring: MLP concatenates scene vector + fragment embedding
            - Relation scoring: MLP computes membership logits from relative features

    Training: full NxN membership matrix, expected loss weighted by P_anchor.
    Joint optimization: both backbone and heads receive gradients.
    """

    def __init__(
        self,
        model_dim,
        anchor_drop_rate,
        relation_drop_rate,
        attention_heads=8,
        attention_blocks=4,
        **kwargs,
    ):
        super().__init__()

        # PeelerBackbone: MLP → self-attention → max pool
        self.backbone = PeelerBackbone(
            model_dim,
            attention_heads=attention_heads,
            attention_blocks=attention_blocks,
        )

        # PeelerLoop: single-fragment iteration (anchor scoring + relation scoring)
        self.peeler_loop = PeelerLoop(model_dim, anchor_drop_rate, relation_drop_rate)

    def forward(self, embeddings, transforms, mask):
        """Forward pass (softmax all the way through).

        Args:
            embeddings: (B, N, 256) - fragment embeddings
            transforms: (B, N, 16) - fragment transforms
            mask: (B, N) - 1 for real fragments

        Returns:
            anchor_probs: (B, N) - softmax distribution over anchors
            affinity_logits: (B, N, N) - raw relation head logits for ALL pairs
        """
        B = int(transforms.shape[0])
        N = int(transforms.shape[1])

        # Step 1: Run backbone once to get scene vector
        scene_vec = self.backbone(transforms)  # (B, 1, model_dim)

        # Step 2: Compute all anchor scores and NxN membership logits in one pass
        anchor_logits, affinity_logits = self.peeler_loop(scene_vec, transforms, embeddings, mask)

        # Apply masking
        anchor_logits = anchor_logits + (1 - mask) * -1e9  # mask padding before softmax
        anchor_probs = torch.softmax(anchor_logits, dim=1)  # (B, N)

        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        affinity_logits = affinity_logits + (1 - mask_2d) * -1e9

        return anchor_probs, affinity_logits
