"""Adaptive Object Peeler - Two-model ONNX architecture.

Architecture:
    Model 1 - PeelerBackbone: MLP projection → self-attention
        Input: transforms(N, 16)
        Output: transformer_out(N, feat_dim) — per-fragment with global context
    
    Model 2 - PeelerLoop: anchor scoring + membership logits
        Input: transformer_out(N,feat_dim), transforms(N,16), mask(N)
        Output: anchor_score, membership_logits(N)
    
    Model 3 - Peeler (joint training): combines both models
        - PeelerBackbone runs once to get transformer_out
        - PeelerLoop computes anchor scores and NxN membership logits

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

    # Normalize by average distance per batch
    avg_distance = torch.norm(translation, dim=-1).mean(dim=1, keepdim=True)  # (B, 1)
    avg_distance = torch.clamp(avg_distance, min=1e-8)
    avg_distance = avg_distance.unsqueeze(-1)  # (B, 1, 1)
    translation = translation / avg_distance
    scale = scale / avg_distance

    return translation, scale, rot

def _compute_relative_features(seed_T, seed_S, cand_T, cand_S):
    """Compute relative features between seed and candidate poses.

    Args:
        seed_T: (B, S, 3) f32 - seed translations
        seed_S: (B, S) f32 - seed scales
        cand_T: (B, N, 3) f32 - candidate translations
        cand_S: (B, N) f32 - candidate scales

    Returns:
        rel_features: (B, S, N, 7) f32 - [dist(1), dist_bwn(1), dist_norm_s(1), dist_norm_c(1), seed_S_log(1), cand_S_log(1), rel_scale(1)]
    """
    diff = cand_T.unsqueeze(1) - seed_T.unsqueeze(2)  # (B, S, N, 3)

    B, S, N, _ = diff.size()

    dist_raw = torch.norm(diff, dim=-1, keepdim=True)  # (B, S, N, 1)

    # Normalize distance by seed scale (division in linear space)
    seed_S_exp = seed_S.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
    cand_S_exp = cand_S.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)

    dist_normalized_s = dist_raw / seed_S_exp
    dist_normalized_s = torch.log10(torch.clamp(dist_normalized_s, min=1e-8))
    dist_normalized_s = (dist_normalized_s) / 8

    dist_normalized_c = dist_raw / cand_S_exp
    dist_normalized_c = torch.log10(torch.clamp(dist_normalized_c, min=1e-8))
    dist_normalized_c = (dist_normalized_c) / 8

    # Distance between centroids taking them as spheres
    dist_bwn = torch.clamp(dist_raw - seed_S_exp - cand_S_exp, min=1e-8)

    dist_bwn_normalized_s = dist_bwn / seed_S_exp
    dist_bwn_normalized_s = torch.log10(torch.clamp(dist_bwn_normalized_s, min=1e-8))
    dist_bwn_normalized_s = (dist_bwn_normalized_s) / 8

    dist_bwn_normalized_c = dist_bwn / cand_S_exp
    dist_bwn_normalized_c = torch.log10(torch.clamp(dist_bwn_normalized_c, min=1e-8))
    dist_bwn_normalized_c = (dist_bwn_normalized_c) / 8

    dist_bwn = torch.log10(dist_bwn) / 8 + 1



    dist = torch.log10(torch.clamp(dist_raw, min=1e-8))
    dist = dist / 8 + 1

    # Log-scale ratio
    rel_scale = torch.log10(cand_S_exp / seed_S_exp) / 8  # (B, S, N, 1)

    seed_S_log = torch.log10(seed_S_exp.expand(-1, -1, N, -1)) / 6
    cand_S_log = torch.log10(cand_S_exp.expand(-1, S, -1, -1)) / 6

    # Normalized direction
    direction = torch.where(dist_raw > 1e-8, diff / dist_raw, torch.zeros_like(diff))  # (B, S, N, 3)

    return torch.cat([direction, dist, dist_bwn, dist_bwn_normalized_s, dist_bwn_normalized_c, dist_normalized_s, dist_normalized_c, seed_S_log, cand_S_log, rel_scale], dim=-1)  # (B, S, N, 7)

# Output dimension of _compute_relative_features
_REL_FEATURE_DIM = 12  # dist + dist_bwn + dist_norm_s + dist_norm_c + seed_S_log + cand_S_log + rel_scale


class SimpleAttentionBlock(nn.Module):
    """ONNX-friendly transformer block with explicit Q/K/V self-attention.

    Uses explicit projections instead of nn.MultiheadAttention to avoid
    hardcoding sequence lengths during ONNX export.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Explicit Q/K/V projections (avoids nn.MultiheadAttention tracing issues)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x, spatial_bias=None):
        B, N, D = x.shape
        q = self.q_proj(self.norm1(x))
        k = self.k_proj(self.norm1(x))
        v = self.v_proj(self.norm1(x))

        # (B, N, D) → (B, N, num_heads, head_dim) → (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention logits [B, H, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Inject pairwise relational spatial bias directly into attention logits
        if spatial_bias is not None:
            attn = attn + spatial_bias

        attn = torch.softmax(attn, dim=-1)

        # Aggregate values and project back
        attn_output = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.out_proj(attn_output)

        # Residual connection
        x = x + q.transpose(1, 2).reshape(B, N, D)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class PeelerBackbone(nn.Module):
    """Embedding projection → self-attention backbone with pairwise relational bias.

    Input: embeddings(N, embed_dim), transforms(N, 16), mask(N)
    Output: transformer_out(N, feat_dim) — each fragment with global context via self-attention
    """

    def __init__(
        self,
        feat_dim,
        embed_dim,
        attention_heads,
        attention_blocks,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.num_blocks = attention_blocks

        # Project pose features (5D) to feat_dim
        self.proj = nn.Sequential(
            nn.Linear(6, self.feat_dim // 4),
            nn.GELU(),
            nn.Linear(self.feat_dim // 4, self.feat_dim * 2),
            nn.GELU(),
            nn.Linear(self.feat_dim * 2, self.feat_dim)
        )

        self.bias_count = self.attention_heads * self.num_blocks

        # Pairwise relational bias generator: 7 features → per-head bias
        self.rel_bias_generator = nn.Sequential(
            nn.Linear(_REL_FEATURE_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, self.bias_count)
        )

        self.blocks = nn.ModuleList([
            SimpleAttentionBlock(self.feat_dim, attention_heads)
            for _ in range(attention_blocks)
        ])
        self.norm = nn.LayerNorm(self.feat_dim)

    def forward(self, transforms, mask):
        """Forward pass for backbone.

        Args:
            transforms: (B, N, 16) - fragment transforms
            mask: (B, N) - 1 for real fragments

        Returns:
            transformer_out: (B, N, feat_dim) — per-fragment representations with global context
        """

        # Extract raw translation and scale
        translation, scale, _ = _transforms_to_pose(transforms)

        # Bounding box center of translations
        min_translation = translation.amin(dim=1, keepdim=True)
        max_translation = translation.amax(dim=1, keepdim=True)
        bbox_center = (min_translation + max_translation) / 2

        # Relative position from bbox center
        relative_translation = translation - torch.mean(translation)
        distance = torch.norm(relative_translation, dim=-1, keepdim=True)

        norm_dist = torch.log10(torch.clamp(distance / scale, min=1e-8)) / 7

        # print(torch.max(norm_dist), " - ", torch.min(norm_dist))

        dir = torch.where(distance > 1e-8, relative_translation / distance, torch.zeros_like(relative_translation))
        distance = (torch.log10(torch.clamp(distance, min=1e-3)) + 3) / 4
        norm_scale = (torch.log10(scale)) / 6

        # Pose features: [relative_xyz(3) + scale(1) + distance(1)] = 5D
        pose_features = torch.cat([dir, norm_scale, distance, norm_dist], dim=-1)  # (B, N, 5)

        # Project pose features to feat_dim
        x = self.proj(pose_features)  # (B, N, 5) → (B, N, feat_dim)

        # ==========================================
        # PAIRWISE RELATIONAL FEATURE EXTRACTION
        # ==========================================
        # Use RAW translation and scale for pairwise features
        seed_T = translation  # (B, N, 3)
        cand_T = translation  # (B, N, 3)
        seed_S = scale.squeeze(-1)  # (B, N)
        cand_S = scale.squeeze(-1)  # (B, N)

        # 7D pairwise relational tensor: (B, N, N, 7)
        pairwise_feats = _compute_relative_features(seed_T, seed_S, cand_T, cand_S)

        # Generate the Relational Attention Bias Map: [B, N, N, 7] → [B, H*L, N, N]
        spatial_bias = self.rel_bias_generator(pairwise_feats)  # (B, N, N, H*L)
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)  # (B, H*L, N, N)
        spatial_bias = spatial_bias.chunk(self.num_blocks, dim=1)  # [L] each (B, H, N, N)

        # Add mask bias so padded positions don't attend
        mask_bias = (1 - mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(1) * -1e9  # (B, 1, N, N)

        # Run Attention Blocks with explicit relational context
        for i, block in enumerate(self.blocks):
            block_bias = spatial_bias[i] + mask_bias
            x = block(x, spatial_bias=block_bias)

        return self.norm(x)  # (B, N, feat_dim)


class PeelerLoop(nn.Module):
    """Anchor scoring + membership logits with ONNX-optimized export path.

    Training: full NxN membership matrix for gradient flow to all anchors.
    ONNX Export: only computes 1×N for best anchor (avoids NxN computation).

    Input: transformer_out(N,feat_dim), transforms(N,16), mask(N)
    Output: anchor_scores(N), membership_logits(N) [export] or (N,N) [training]
    """

    def __init__(self, feat_dim, anchor_proj_dim, rel_hidden_dim, anchor_drop_rate, relation_drop_rate):
        super().__init__()
        self.feat_dim = feat_dim
        self.anchor_proj_dim = anchor_proj_dim
        self.rel_hidden_dim = rel_hidden_dim

        # Anchor head: simple linear scorer on transformer output
        # transformer_out already contains global context (self-attention) + local pose info
        # High scores -> complex, identifiable parts (receiver, barrel)
        # Low scores -> simple, redundant parts (screws, noise)
        self.anchor_score_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 4),
            nn.GELU(),
            nn.Dropout(anchor_drop_rate),
            nn.Linear(self.feat_dim * 4, self.feat_dim),
            nn.GELU(),
            nn.Dropout(anchor_drop_rate),
            nn.Linear(self.feat_dim, self.feat_dim // 4),
            nn.GELU(),
            nn.Dropout(anchor_drop_rate),
            nn.Linear(self.feat_dim // 4, 1),
        )

        # Relation head: project transformer output to small dim, concat with relative features
        self.anchor_rel_proj = nn.Linear(self.feat_dim, self.anchor_proj_dim)

        # Scene-relative projection: concatenates projected anchor with relative features, projects to 32D
        self.rel_proj_mlp = nn.Sequential(
            nn.Linear(self.anchor_proj_dim + _REL_FEATURE_DIM, self.rel_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(relation_drop_rate),
            nn.Linear(self.rel_hidden_dim * 2, self.rel_hidden_dim),
            nn.GELU(),
            nn.Dropout(relation_drop_rate),
            nn.Linear(self.rel_hidden_dim, self.rel_hidden_dim),
            nn.GELU(),
            nn.Linear(self.rel_hidden_dim, self.rel_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(relation_drop_rate),
            nn.Linear(self.rel_hidden_dim // 2, 1),
        )

    def forward(self, transformer_out, transforms, embeddings=None, mask=None):
        """Forward pass with ONNX-optimized export path.

        Args:
            transformer_out: (B, N, feat_dim) - per-fragment representations with global context
            transforms: (B, N, 16) - fragment transforms
            embeddings: (B, N, 256) - all fragment embeddings (not used)
            mask: (B, N) - 1 for real fragments (training only)

        Returns:
            anchor_scores: (B, N) - anchor logits for all fragments
            membership_logits: (B, N) for ONNX export, (B, N, N) for training
        """
        B, N, _ = transforms.shape

        # Anchor head: score ALL fragments as anchors in parallel
        # transformer_out already contains global context (self-attention) + local pose info
        anchor_scores = self.anchor_score_head(transformer_out).squeeze(-1)  # (B, N)

        if torch.onnx.is_in_onnx_export():
            # ONNX export: only compute affinities for top anchor (avoids NxN)
            seed_idx = torch.argmax(anchor_scores, dim=1)  # (B,)

            # Extract pose features for ONNX path
            translation, scale, rot = _transforms_to_pose(transforms)

            # Gather seed translation and scale
            seed_idx_T = seed_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3)  # (B, 1, 3)
            seed_T = torch.gather(translation, 1, seed_idx_T)  # (B, 1, 3)
            seed_idx_S = seed_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, scale.size(2))  # (B, 1, 1)
            seed_S = torch.gather(scale, 1, seed_idx_S).squeeze(1)  # (B, 1, 1) -> (B, 1)

            # Compute relative features: seed vs all N candidates
            rel_features = _compute_relative_features(seed_T, seed_S, translation, scale.squeeze(-1))  # (B, 1, N, 7)

            # Project anchor + relative features to 32D
            anchor_proj = self.anchor_rel_proj(transformer_out)  # (B, N, anchor_proj_dim)
            # Gather anchor projection for best anchor: (B, N, anchor_proj_dim) → (B, 1, anchor_proj_dim)
            gather_indices = seed_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.anchor_proj_dim)  # (B, 1, anchor_proj_dim)
            anchor_proj_seed = torch.gather(anchor_proj, 1, gather_indices)  # (B, 1, anchor_proj_dim)
            anchor_proj_seed = anchor_proj_seed.unsqueeze(2).expand(-1, -1, N, -1)  # (B, 1, N, anchor_proj_dim)
            rel_context = torch.cat([anchor_proj_seed, rel_features], dim=-1)  # (B, 1, N, anchor_proj_dim + 6)
            rel_projected = self.rel_proj_mlp(rel_context)  # (B, 1, N, 32)

            # Relation head → (B, 1, N, 1) → (B, 1, N) → (B, N)
            membership_logits = rel_projected.squeeze(-1)  # (B, 1, N, 1) → (B, 1, N)
            membership_logits = membership_logits.squeeze(1)  # (B, 1, N) → (B, N)
            membership_logits = membership_logits + (1 - mask) * -1e9
        else:
            # Training: full NxN for gradient flow to all anchors
            translation, scale, rot = _transforms_to_pose(transforms)
            rel_features = _compute_relative_features(translation, scale.view(B, N), translation, scale.squeeze(-1))  # (B, N, N, 7)

            # Project anchor + relative features to 32D
            anchor_proj = self.anchor_rel_proj(transformer_out)  # (B, N, anchor_proj_dim)
            anchor_proj = anchor_proj.unsqueeze(2)  # (B, N, 1, anchor_proj_dim)
            rel_context = torch.cat([anchor_proj.expand(-1, -1, N, -1), rel_features], dim=-1)  # (B, N, N, anchor_proj_dim + 6)
            rel_projected = self.rel_proj_mlp(rel_context)  # (B, N, N, 32)

            # Relation head → (B, N, N)
            membership_logits = rel_projected.squeeze(-1)  # (B, N, N)

            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
            membership_logits = membership_logits + (1 - mask_2d) * -1e9

        return anchor_scores, membership_logits


@MODELS.register_module()
class Peeler(nn.Module):
    """Adaptive Object Peeler model (joint training).

    Full forward pass (softmax all the way through):
        1. PeelerBackbone: MLP → self-attention → per-fragment representations
        2. PeelerLoop: for each fragment as anchor:
            - Anchor scoring: linear scorer on transformer output
            - Relation scoring: MLP computes membership logits from projected anchor + relative features

    Training: full NxN membership matrix, expected loss weighted by P_anchor.
    Joint optimization: both backbone and heads receive gradients.
    """

    def __init__(
        self,
        feat_dim,
        embed_dim,
        anchor_proj_dim,
        rel_hidden_dim,
        anchor_drop_rate,
        relation_drop_rate,
        attention_heads,
        attention_blocks,
        **kwargs,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.anchor_proj_dim = anchor_proj_dim
        self.rel_hidden_dim = rel_hidden_dim

        # PeelerBackbone: embedding projection → self-attention with pairwise relational bias
        self.backbone = PeelerBackbone(
            feat_dim,
            embed_dim,
            attention_heads=attention_heads,
            attention_blocks=attention_blocks,
        )

        # PeelerLoop: single-fragment iteration (anchor scoring + relation scoring)
        self.peeler_loop = PeelerLoop(feat_dim, anchor_proj_dim, rel_hidden_dim, anchor_drop_rate, relation_drop_rate)

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

        # Step 1: Run backbone once to get per-fragment representations with global context
        transformer_out = self.backbone(transforms, mask)  # (B, N, feat_dim)

        # Step 2: Compute all anchor scores and NxN membership logits in one pass
        anchor_logits, affinity_logits = self.peeler_loop(transformer_out, transforms, embeddings, mask)

        # Apply masking
        anchor_logits = anchor_logits + (1 - mask) * -1e9  # mask padding before softmax
        anchor_probs = torch.softmax(anchor_logits, dim=1)  # (B, N)

        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        affinity_logits = affinity_logits + (1 - mask_2d) * -1e9

        return anchor_probs, affinity_logits
