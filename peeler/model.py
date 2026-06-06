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




def _transforms_to_pose(transforms):
    """Extract translation, scale, and normalized rotation from 4x4 transforms.

    Args:
        transforms: (B, N, 16) f32 - row-major 4x4 matrices

    Returns:
        translation: (B, N, 3)
        scale: (B, N, 1) clamped to min 1e-8
        rot: (B, N, 9) flattened normalized rotation matrix
    """
    B, N, _ = transforms.shape
    mat = transforms.view(B, N, 4, 4)
    translation = mat[:, :, :3, 3]
    scale = torch.norm(mat[:, :, :3, :3], dim=-1).mean(-1, keepdim=True)
    scale = torch.clamp(scale, min=1e-8)
    rot = mat[:, :, :3, :3].reshape(B, N, -1) / scale
    return translation, scale, rot

# Pose feature dimension: translation(3) + scale(1)
_POSE_DIM = 4


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
    Output: scene_vec(1, feat_dim)
    """

    def __init__(
        self,
        feat_dim,
        attention_heads,
        attention_blocks,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        self.proj = nn.Linear(_POSE_DIM, self.feat_dim)
        self.blocks = nn.ModuleList([
            SimpleAttentionBlock(self.feat_dim, attention_heads)
            for _ in range(attention_blocks)
        ])
        self.norm = nn.LayerNorm(self.feat_dim)

    def forward(self, transforms):
        """Forward pass for backbone.

        Args:
            transforms: (B, N, 16) - fragment transforms

        Returns:
            scene_vec: (B, 1, 16) - global scene vector
        """
        translation, scale, _ = _transforms_to_pose(transforms)
        x = torch.cat([translation, scale], dim=-1)  # (B, N, 4)
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

    Input: scene_vec(1,feat_dim), transforms(N,16), embeddings(N,256), mask(N)
    Output: anchor_scores(N), membership_logits(N) [export] or (N,N) [training]
    """

    def __init__(self, feat_dim, rel_hidden_dim, anchor_drop_rate, relation_drop_rate):
        super().__init__()
        self.feat_dim = feat_dim
        self.rel_hidden_dim = rel_hidden_dim

        # Anchor head: MLP that scores each fragment as a potential anchor seed
        # Uses pose features (translation + scale) instead of embeddings
        # High scores -> complex, identifiable parts (receiver, barrel)
        # Low scores -> simple, redundant parts (screws, noise)
        self.anchor_pose_proj = nn.Linear(_POSE_DIM, self.feat_dim)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim * 4),
            nn.GELU(),
            nn.Dropout(anchor_drop_rate),
            nn.Linear(self.feat_dim * 4, self.feat_dim),
            nn.GELU(),
            nn.Dropout(anchor_drop_rate),
            nn.Linear(self.feat_dim, 1),
        )

        # Relation head: MLP that computes membership logits from relative features
        self.relation_mlp = nn.Sequential(
            nn.Linear(_REL_FEATURE_DIM, self.rel_hidden_dim),
            nn.GELU(),
            nn.Dropout(relation_drop_rate),
            nn.Linear(self.rel_hidden_dim, self.rel_hidden_dim),
            nn.GELU(),
            nn.Linear(self.rel_hidden_dim, self.rel_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(relation_drop_rate),
            nn.Linear(self.rel_hidden_dim // 2, 1),
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

        # Extract pose features (translation + scale) from transforms for anchor head
        translation, scale, rot = _transforms_to_pose(transforms)
        pose_input = torch.cat([translation, scale], dim=-1)  # (B, N, 4)

        # Anchor head: score ALL fragments as anchors in parallel
        pose_proj = self.anchor_pose_proj(pose_input)  # (B, N, feat_dim)
        scene_expanded = scene_vec.expand(-1, N, self.feat_dim)  # (B, N, feat_dim)
        context = torch.cat([scene_expanded, pose_proj], dim=-1)  # (B, N, feat_dim*2)
        anchor_scores = self.anchor_mlp(context).squeeze(-1)  # (B, N)

        if torch.onnx.is_in_onnx_export():
            # ONNX export: only compute affinities for top anchor (avoids NxN)
            seed_idx = torch.argmax(anchor_scores, dim=1)  # (B,)

            # Gather seed translation and scale
            seed_idx_T = seed_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3)  # (B, 1, 3)
            seed_idx_S = seed_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, 1)  # (B, 1, 1)
            seed_T = torch.gather(translation, 1, seed_idx_T)  # (B, 1, 3)
            seed_S = torch.gather(scale, 1, seed_idx_S).squeeze(-1)  # (B, 1)

            # Compute relative features: seed vs all N candidates
            rel_features = self._compute_relative_features(seed_T, seed_S, translation, scale.squeeze(-1))  # (B, 1, N, 6)

            # Relation head → (B, 1, N, 1) → (B, 1, N) → (B, N)
            membership_logits = self.relation_mlp(rel_features).squeeze(-1)  # (B, 1, N, 1) → (B, 1, N)
            membership_logits = membership_logits.squeeze(1)  # (B, 1, N) → (B, N)
            membership_logits = membership_logits + (1 - mask) * -1e9
        else:
            # Training: full NxN for gradient flow to all anchors
            rel_features = self._compute_relative_features(translation, scale.view(B, N), translation, scale.squeeze(-1))  # (B, N, N, 6)

            # Relation head → (B, N, N)
            membership_logits = self.relation_mlp(rel_features).squeeze(-1)  # (B, N, N)

            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
            membership_logits = membership_logits + (1 - mask_2d) * -1e9

        return anchor_scores, membership_logits

    def _compute_relative_features(self, seed_T, seed_S, cand_T, cand_S):
        """Compute relative features between seed and candidate poses.

        Args:
            seed_T: (B, S, 3) f32 - seed translations
            seed_S: (B, S) f32 - seed scales
            cand_T: (B, N, 3) f32 - candidate translations
            cand_S: (B, N) f32 - candidate scales

        Returns:
            rel_features: (B, S, N, 6) f32 - [dist(1), dist_norm_s(1), dist_norm_c(1), seed_S_log(1), cand_S_log(1), rel_scale(1)]
        """
        diff = cand_T.unsqueeze(1) - seed_T.unsqueeze(2)  # (B, S, N, 3)

        B, S, N, _ = diff.size()

        dist_raw = torch.norm(diff, dim=-1, keepdim=True)  # (B, S, N, 1)
        
        # Normalize distance by seed scale (division in linear space)
        seed_S_exp = seed_S.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
        cand_S_exp = cand_S.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)

        dist_normalized_s = dist_raw / seed_S_exp
        dist_normalized_s = torch.log10(dist_normalized_s + 1e-8)

        dist_normalized_c = dist_raw / cand_S_exp
        dist_normalized_c = torch.log10(dist_normalized_c + 1e-8)
        
        dist = torch.log10(dist_raw + 1e-8)
        
        # Log-scale ratio
        rel_scale = torch.log10(cand_S_exp / seed_S_exp)  # (B, S, N, 1)

        seed_S_log = torch.log10(seed_S_exp.expand(-1, -1, N, -1))
        cand_S_log = torch.log10(cand_S_exp.expand(-1, S, -1, -1))
        
        # Normalized direction
        direction = diff / (dist_raw + 1e-8)  # (B, S, N, 3)

        return torch.cat([dist, dist_normalized_s, dist_normalized_c, seed_S_log, cand_S_log, rel_scale], dim=-1)  # (B, S, N, 6)


# Output dimension of _compute_relative_features
_REL_FEATURE_DIM = 6  # dist + dist_norm_s + dist_norm_c + seed_S_log + cand_S_log + rel_scale


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
        feat_dim,
        rel_hidden_dim,
        anchor_drop_rate,
        relation_drop_rate,
        attention_heads,
        attention_blocks,
        **kwargs,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.rel_hidden_dim = rel_hidden_dim

        # PeelerBackbone: MLP → self-attention → max pool
        self.backbone = PeelerBackbone(
            feat_dim,
            attention_heads=attention_heads,
            attention_blocks=attention_blocks,
        )

        # PeelerLoop: single-fragment iteration (anchor scoring + relation scoring)
        self.peeler_loop = PeelerLoop(feat_dim, rel_hidden_dim, anchor_drop_rate, relation_drop_rate)

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
        scene_vec = self.backbone(transforms)  # (B, 1, feat_dim)

        # Step 2: Compute all anchor scores and NxN membership logits in one pass
        anchor_logits, affinity_logits = self.peeler_loop(scene_vec, transforms, embeddings, mask)

        # Apply masking
        anchor_logits = anchor_logits + (1 - mask) * -1e9  # mask padding before softmax
        anchor_probs = torch.softmax(anchor_logits, dim=1)  # (B, N)

        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        affinity_logits = affinity_logits + (1 - mask_2d) * -1e9

        return anchor_probs, affinity_logits
