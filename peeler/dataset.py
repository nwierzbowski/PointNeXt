"""PeelerDataset - All-gold soup generation for adaptive object peeler training.

Takes per-mesh TBO data (272 channels: 256 emb + 16 transform per fragment)
and generates soups by mixing K random assets together. Every fragment in
the soup belongs to a real asset. Y encodes same-asset membership for all
assets in the soup.

Registered with openpoints DATASETS registry.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from openpoints.dataset.build import DATASETS
from .training.augmentation import AugmentationEngine


@DATASETS.register_module()
class PeelerDataset(Dataset):
    """All-gold soup dataset for peeler training.

    Generates training samples by mixing K random assets together.
    Each sample is a "soup" of N fragments with an N x N ground truth
    matrix Y where Y_ij=1 if fragments i and j belong to the same asset.

    Args:
        all_embeddings: list of numpy arrays, one per asset, shape (N_i, 256)
        all_transforms: list of numpy arrays, one per asset, shape (N_i, 16)
        max_fragments: int, maximum total fragments per soup
        seed: int, random seed for reproducibility
        translation_scale: tuple (mean, std) for lognormal translation noise
        asset_scale_std: float or tuple for per-asset scale augmentation
        scene_scale: tuple (low, high) for uniform scene scale augmentation
        cluster_translation_scale: tuple (mean, std) for cluster spacing
        embedding_noise_sigma: float, std dev of Gaussian noise added to embeddings (0 = disabled)
        translation_noise_sigma: float, std dev of per-fragment Gaussian noise on translation columns (0 = disabled)
        scaling_noise_sigma: float, std dev of per-fragment Gaussian noise on rotation/scale block (0 = disabled)
        per_asset_rotation: bool, apply uniform random SO(3) rotation to each asset's fragments
    """

    def __init__(
        self,
        all_embeddings,
        all_transforms,
        max_fragments,
        seed,
        translation_scale,
        asset_scale_std,
        scene_scale,
        cluster_translation_scale,
        scene_transforms,
        embedding_noise_sigma=0.0,
        translation_noise_sigma=0.0,
        scaling_noise_sigma=0.0,
        per_asset_rotation=True,
    ):
        self.all_embeddings = all_embeddings  # list of (N_i, 256)
        self.all_transforms = all_transforms  # list of (N_i, 16)
        self.scene_transforms = scene_transforms  # list of (N_i, 16) from transforms TBO
        self.max_fragments = max_fragments
        self.embedding_noise_sigma = embedding_noise_sigma
        self.seed = seed
        self._epoch = 0

        self.n_assets = len(self.all_embeddings)

        # Augmentation engine for random soup transforms
        self._engine = AugmentationEngine(
            translation_noise_sigma=translation_noise_sigma,
            scaling_noise_sigma=scaling_noise_sigma,
            per_asset_rotation=per_asset_rotation,
            asset_scale_std=asset_scale_std,
            translation_scale=translation_scale,
            cluster_translation_scale=cluster_translation_scale,
            scene_scale=scene_scale,
        )

        # Build fragment count buckets (cap oversized assets into max_fragments bucket)
        self._fragments_by_count = {}
        for i in range(self.n_assets):
            n = len(self.all_embeddings[i])
            effective_n = min(n, self.max_fragments)
            self._fragments_by_count.setdefault(effective_n, []).append(i)

        self._bucket_sizes = np.array(sorted(self._fragments_by_count.keys()), dtype=np.int32)

    def set_epoch(self, epoch: int):
        """Set the epoch for randomization."""
        self._epoch = epoch

    @staticmethod
    def _sample_soup_partition(rng, N, mu=None, u=None):
        """Capped GMS sampler for perfect pairwise matrix uniformity.

        Caps the maximum asset count to preserve physical asset sizes (no singletons),
        while using GMS to guarantee that the actual connection density (mu)
        and relational entropy (u) are perfectly uniform with zero rounding collapse.
        """
        # Set the maximum number of assets allowed in a soup
        # (With N=150, capping at 30 guarantees average asset size >= 5 fragments)
        if mu is None and u is None:
            if rng.uniform(0.0, 1.0) < 0.02:
                return [N], 1, float('inf'), 1.0, 0.0

        K_cap = max(N // 6, 2)

        # Sample target density mu uniformly above the K_cap limit
        if mu is None:
            r = rng.uniform(0.0, 1.0)
            mu = 1.0 / K_cap + r * (1.0 - 1.0 / K_cap)
        mu = np.clip(mu, 1.0 / K_cap, 1.0)

        if np.isclose(mu, 1.0):
            return [N], 1, float('inf'), mu, 0.0

        if u is None:
            u = rng.uniform(0.0, 1.0)
        u = np.clip(u, 0.0, 1.0)

        # Interpolate k between k_min and K_cap
        k_min = int(np.ceil(1.0 / mu))
        k_max = K_cap
        k = int(np.round(k_min + u * (k_max - k_min)))
        k = max(2, min(k, K_cap))

        # 2. CALIBRATION: Invert the GMS scaling offset
        # This solves for the exact weight density needed to yield a final mu
        # after the +1 upfront allocation is added back.
        free_budget = N - k
        if free_budget > 0:
            mu_weights = (mu * N * N - 2 * N + k) / (free_budget * free_budget)
            mu_weights = np.clip(mu_weights, 1.0 / k, 1.0)
        else:
            mu_weights = 1.0 / k

        denominator = (mu_weights * k) - 1.0
        if np.isclose(denominator, 0.0) or denominator < 0:
            alpha = 10000.0
        else:
            alpha = (1.0 - mu_weights) / denominator
            alpha = max(alpha, 0.01)

        weights = rng.dirichlet(np.ones(k) * alpha)

        # --- GUARANTEED MINIMUM SIZE (GMS) ALLOCATION ---
        # Distribute N fragments among k assets, guaranteeing size >= 1
        free_budget = N - k
        
        if free_budget > 0:
            scaled = weights * free_budget
            sizes = np.floor(scaled).astype(int)

            remainder = free_budget - np.sum(sizes)
            if remainder > 0:
                fractional_parts = scaled - sizes
                largest_indices = np.argsort(fractional_parts)[::-1][:remainder]
                sizes[largest_indices] += 1
            
            final_sizes = sizes + 1
        else:
            final_sizes = np.ones(k, dtype=np.int32)

        if len(final_sizes) == 0:
            final_sizes = [N]
            k = 1
            alpha = float('inf')

        return sorted(list(final_sizes), reverse=True), k, alpha, mu, u

    def __len__(self):
        return self.n_assets

    def _find_bucket_at_or_below(self, target_size):
        """Find the bucket size at or below target_size. Falls back to smallest bucket."""
        idx = np.searchsorted(self._bucket_sizes, target_size, side='right') - 1
        if idx < 0:
            idx = 0
        return int(self._bucket_sizes[idx])

    def __getitem__(self, idx):
        # Create fresh RNG seeded by epoch + idx for full randomization
        rng = np.random.RandomState(self.seed + idx + self._epoch * 100000)

        # N = int(rng.randint(2, self.max_fragments + 1))
        N = self.max_fragments
        sizes, k, alpha_cluster_prob, mu, u = self._sample_soup_partition(rng, N)

        soup_emb_list = []
        soup_trans_list = []
        asset_ids_list = []
        orig_indices_list = []
        total_fragments = 0
        asset_fragments = []
        soup_asset_gids = []

        for s in sizes:
            bucket_size = self._find_bucket_at_or_below(s)
            bucket_pool = self._fragments_by_count[bucket_size]
            asset_gid = int(rng.choice(bucket_pool))

            emb = self.all_embeddings[asset_gid]
            trans = self.all_transforms[asset_gid]
            n_available = len(emb)

            # Sample bucket_size fragments from the asset (handles oversized assets)
            if n_available > bucket_size:
                chosen_idx = rng.choice(n_available, bucket_size, replace=False)
                chosen_idx.sort()
                emb = emb[chosen_idx]
                trans = trans[chosen_idx]
                local_indices = chosen_idx
            else:
                local_indices = np.arange(n_available)

            n_fragments = len(emb)

            soup_emb_list.append(emb)
            soup_trans_list.append(trans)
            asset_ids_list.append(np.full(n_fragments, len(asset_ids_list), dtype=np.int64))
            orig_indices_list.extend(local_indices + asset_gid * 1_000_000_000)

            total_fragments += n_fragments
            asset_fragments.append(n_fragments)
            soup_asset_gids.append(asset_gid)
        soup_emb = np.concatenate(soup_emb_list, axis=0)
        soup_trans = np.concatenate(soup_trans_list, axis=0)
        asset_ids = np.concatenate(asset_ids_list, axis=0)
        orig_indices = np.array(orig_indices_list, dtype=np.int64)

        # Apply random augmentation to transforms
        self._engine.apply_full_random(soup_trans, asset_fragments, soup_asset_gids, rng, k, self.scene_transforms)

        # Shuffle soup
        shuffle_idx = rng.permutation(len(soup_emb))
        soup_emb = soup_emb[shuffle_idx]
        if self.embedding_noise_sigma > 0:
            soup_emb = soup_emb + np.random.default_rng(self.seed + self._epoch).normal(scale=self.embedding_noise_sigma, size=soup_emb.shape).astype(soup_emb.dtype)
        soup_trans = soup_trans[shuffle_idx]
        asset_ids = asset_ids[shuffle_idx]
        orig_indices = orig_indices[shuffle_idx]

        # Build N x N ground truth: Y_ij = 1 if same asset (bool, 1/4 memory of float32)
        Y = (asset_ids[:, None] == asset_ids[None, :])

        actual_n = len(soup_emb)
        actual_mu = sum(s * s for s in asset_fragments) / (actual_n * actual_n) if actual_n > 0 else 0.0
        actual_k = len(asset_fragments)
        k_min = int(np.ceil(1.0 / actual_mu))
        k_max = max(N // 6, 2)
        actual_u = (actual_k - k_min) / (k_max - k_min) if (k_max - k_min) > 0 else 0.0
        actual_u = float(np.clip(actual_u, 0.0, 1.0))

        return {
            'embeddings': torch.from_numpy(soup_emb),
            'transforms': torch.from_numpy(soup_trans),
            'Y': torch.from_numpy(Y),
            'orig_indices': torch.from_numpy(orig_indices),
            'asset_ids': torch.from_numpy(asset_ids),
            'soup_count': len(asset_fragments),
            'soup_stats': {
                'num_assets': len(asset_fragments),
                'avg_fragments': sum(asset_fragments) / len(asset_fragments) if asset_fragments else 0,
                'asset_fragments': list(asset_fragments),
                'target_mu': mu,
                'target_u': u,
                'actual_mu': actual_mu,
                'actual_u': actual_u,
                'k': k,
                'alpha': alpha_cluster_prob,
                'target_n': N,
                'actual_n': actual_n,
            },
        }

    def compute_stats(self, num_samples=None):
        """Compute dataset statistics by iterating samples once.

        Args:
            num_samples: Number of samples to process. None = entire dataset.

        Returns:
            (asset_counts, asset_fragments, actual_ns, actual_mus, actual_us) tuples for histogram widget
        """
        asset_counts = []
        asset_fragments = []
        actual_ns = []
        actual_mus = []
        actual_us = []
        n = num_samples or len(self)
        for i in range(n):
            sample = self[i]
            stats = sample.get('soup_stats', {})
            if stats:
                asset_counts.append(stats['num_assets'])
                asset_fragments.append(tuple(stats['asset_fragments']))
                actual_ns.append(stats['actual_n'])
                actual_mus.append(stats['actual_mu'])
                actual_us.append(stats['actual_u'])
        return asset_counts, asset_fragments, actual_ns, actual_mus, actual_us

    @staticmethod
    def collate_fn(datas):
        """Collate function for PeelerDataset.

        Pads samples to the longest soup in the batch.
        """
        max_n = max(len(d['embeddings']) for d in datas)

        embeddings_list = []
        transforms_list = []
        Y_list = []
        orig_indices_list = []
        asset_ids_list = []
        mask_list = []

        # Pre-create a flat identity matrix for padding [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        identity_flat = torch.eye(4).view(16)

        # Pre-create an identity-like embedding pattern to avoid degenerate zero vectors.
        # Uses a sparse pattern: first dim=1, rest=0, repeated to fill 256 dims.
        # This gives the FeatureLift a structured (non-zero) input for padding nodes.
        embedding_dim = 256
        identity_emb = torch.zeros(1, embedding_dim, dtype=torch.float32)
        identity_emb[0, 0] = 1.0

        for d in datas:
            n = len(d['embeddings'])
            pad_size = max_n - n

            mask =  torch.cat([torch.ones(n), torch.zeros(pad_size)])
            mask_list.append(mask)

            if pad_size > 0:
                embeddings_list.append(
                    torch.cat([
                        d['embeddings'],
                        identity_emb.repeat(pad_size, 1),
                    ], dim=0)
                )
                transforms_list.append(
                    torch.cat([
                        d['transforms'],
                        identity_flat.repeat(pad_size, 1),
                    ], dim=0)
                )
                Y_list.append(
                    torch.cat([
                        torch.cat([d['Y'], torch.zeros(n, pad_size, dtype=torch.bool)], dim=1),
                        torch.zeros(pad_size, max_n, dtype=torch.bool),
                    ], dim=0)
                )
                orig_indices_list.append(
                    torch.cat([
                        d['orig_indices'],
                        torch.full((pad_size,), -999, dtype=torch.int64),
                    ], dim=0)
                )
                asset_ids_list.append(
                    torch.cat([
                        d['asset_ids'],
                        torch.full((pad_size,), -1, dtype=torch.int64),
                    ], dim=0)
                )
            else:
                embeddings_list.append(d['embeddings'])
                transforms_list.append(d['transforms'])
                Y_list.append(d['Y'])
                orig_indices_list.append(d['orig_indices'])
                asset_ids_list.append(d['asset_ids'])

        return {
            'embeddings': torch.stack(embeddings_list),
            'transforms': torch.stack(transforms_list),
            'Y': torch.stack(Y_list),
            'orig_indices': torch.stack(orig_indices_list),
            'asset_ids': torch.stack(asset_ids_list),
            'mask': torch.stack(mask_list), # [B, N]
            'soup_count': torch.tensor([d['soup_count'] for d in datas], dtype=torch.float32),
            'soup_stats': [d['soup_stats'] for d in datas],
        }
