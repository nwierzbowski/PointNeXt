from torch import nn
import torch

from peeler.model import _REL_FEATURE_DIM, TRANSFORM_DIM, compute_relative_features, compute_transform_features, transforms_to_pose


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
        identity = x

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
        x = x + identity

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class PeelerBackbone(nn.Module):
    """Embedding projection → self-attention backbone with pairwise relational bias.

    Input: transforms(N, 16), mask(N), embeddings(N, embed_dim)
    Output: (B, N, feat_dim) — each fragment with global context via self-attention
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

        # Project pose features (22D) + embeddings (embed_dim) to feat_dim
        self.proj = nn.Sequential(
            nn.Linear(TRANSFORM_DIM, self.feat_dim * 4),
            nn.GELU(),
            nn.Linear(self.feat_dim * 4, self.feat_dim * 2),
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

    

    def forward(self, transforms, mask, embeddings):
        """Forward pass for backbone.

        Args:
            transforms: (B, N, 16) - fragment transforms
            mask: (B, N) - 1 for real fragments
            embeddings: (B, N, embed_dim) - fragment embeddings

        Returns:
            transformer_out: (B, N, feat_dim) — per-fragment representations with global context
        """
        # print(torch.max(norm_dist), " - ", torch.min(norm_dist))
        # Extract raw translation, scale, and rotation
        translation, scale, rot = transforms_to_pose(transforms)

        trans_feats = compute_transform_features(translation, scale)

        # Pose features: [relative_xyz(3) + scale(1) + distance(1) + projected_rotation(16)] = 21D
        pose_features = torch.cat([trans_feats, embeddings], dim=-1)  # (B, N, 22 + embed_dim)

        # Project pose features to feat_dim
        x = self.proj(trans_feats)  # (B, N, 22+embed_dim) → (B, N, feat_dim)

        # ==========================================
        # PAIRWISE RELATIONAL FEATURE EXTRACTION
        # ==========================================
        # Use RAW translation and scale for pairwise features
        seed_T = translation  # (B, N, 3)
        cand_T = translation  # (B, N, 3)
        seed_S = scale.squeeze(-1)  # (B, N)
        cand_S = scale.squeeze(-1)  # (B, N)

        # Pairwise relational tensor with rotation: (B, N, N, 22)
        pairwise_feats = compute_relative_features(seed_T, seed_S, cand_T, cand_S, rot, rot)

        # Generate the Relational Attention Bias Map: [B, N, N, 7] → [B, H*L, N, N]
        spatial_bias = self.rel_bias_generator(pairwise_feats)  # (B, N, N, H*L)
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)  # (B, H*L, N, N)
        spatial_bias = spatial_bias.chunk(self.num_blocks, dim=1)  # [L] each (B, H, N, N)

        # Add mask bias so padded positions don't attend
        mask_bias = (1 - mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(1) * -1e9  # (B, 1, N, N)

        # Run Attention Blocks with explicit relational context
        for i, block in enumerate(self.blocks):
            block_bias = spatial_bias[i] + mask_bias
            x = block(x)

        return self.norm(x)  # (B, N, feat_dim)
