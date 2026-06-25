"""PeelerDataset - All-gold soup generation for adaptive object peeler training.

Takes per-mesh TBO data (272 channels: 256 emb + 16 transform per fragment)
and generates soups by mixing K random assets together. Every fragment in
the soup belongs to a real asset. Y encodes same-asset membership for all
assets in the soup.

Registered with openpoints DATASETS registry.
"""
import numpy as np
import torch
import math
from torch.utils.data import Dataset

from openpoints.dataset.build import DATASETS


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
        cluster_translation_scale
    ):
        self.all_embeddings = all_embeddings  # list of (N_i, 256)
        self.all_transforms = all_transforms  # list of (N_i, 16)
        self.max_fragments = max_fragments
        self.translation_scale = translation_scale
        self.asset_scale_std = asset_scale_std
        self.scene_scale = scene_scale
        self.cluster_translation_scale = cluster_translation_scale
        self.seed = seed
        self._epoch = 0

        self.n_assets = len(self.all_embeddings)

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

    def __len__(self):
        return self.n_assets

    def __getitem__(self, idx):
        # Create fresh RNG seeded by epoch + idx for full randomization
        rng = np.random.RandomState(self.seed + idx + self._epoch * 100000)

        # Dynamic Budget Allocation Loop
        MAX_TOTAL_FRAGMENTS = int(rng.randint(1, self.max_fragments + 1))

        # Dynamic K distribution based on actual budget
        max_k_half = max(1, MAX_TOTAL_FRAGMENTS // 2)
        k_values = list(range(1, max_k_half + 1))
        fixed_weights = [0.1, 0.05, 0.04, 0.03, 0.015]
        
        if max_k_half <= 5:
            k_weights = np.array(fixed_weights[:max_k_half], dtype=np.float32)
        else:
            uniform_weight = 0.765 / (max_k_half - 5)
            k_weights = np.array(fixed_weights + [uniform_weight] * (max_k_half - 5), dtype=np.float32)
        
        k_weights /= k_weights.sum()
        k = int(rng.choice(k_values, p=k_weights))

        soup_emb_list = []
        soup_trans_list = []
        asset_ids_list = []
        orig_indices_list = []
        total_fragments = 0
        asset_fragments = []
        soup_asset_gids = []
        first_asset_fragments = None

        for step in range(k):
            if total_fragments >= MAX_TOTAL_FRAGMENTS:
                break
                
            remaining_slots = k - len(asset_ids_list) - 1  # slots AFTER this one
            remaining_budget = MAX_TOTAL_FRAGMENTS - total_fragments
            max_asset_size = remaining_budget - remaining_slots  # leave at least 1 per remaining slot

            # Log-weighted selection: each doubling range gets equal probability
            max_eligible = max_asset_size
            if first_asset_fragments is not None:
                max_eligible = min(max_eligible, first_asset_fragments)

            idx_limit = np.searchsorted(self._bucket_sizes, max_eligible, side='right')
            if idx_limit == 0:
                break  # no eligible assets, stop early
            
            eligible_sizes = self._bucket_sizes[:idx_limit]
            if first_asset_fragments is None:
                # First asset: uniform per bucket
                weights = np.ones(len(eligible_sizes), dtype=np.float32)
            else:
                # Subsequent assets: log1p(size) directly → favors larger sizes
                weights = np.log1p(eligible_sizes).astype(np.float32)
            weights /= weights.sum()
            
            chosen_idx = rng.choice(len(eligible_sizes), p=weights)
            chosen_bucket_size = int(eligible_sizes[chosen_idx])
            bucket_pool = self._fragments_by_count[chosen_bucket_size]
            asset_gid = int(rng.choice(bucket_pool))
            
            emb = self.all_embeddings[asset_gid]
            trans = self.all_transforms[asset_gid]
            n_fragments = len(emb)

            if first_asset_fragments is None:
                first_asset_fragments = n_fragments

            # Cap the final asset if we overflow the budget
            if total_fragments + n_fragments > MAX_TOTAL_FRAGMENTS:
                n_fragments = MAX_TOTAL_FRAGMENTS - total_fragments
                if n_fragments <= 0:
                    break
                indices = rng.choice(len(emb), size=n_fragments, replace=False)
                indices = np.sort(indices)
                emb = emb[indices]
                trans = trans[indices]
                n_fragments = len(emb)

            soup_emb_list.append(emb)
            soup_trans_list.append(trans)
            asset_ids_list.append(np.full(n_fragments, len(asset_ids_list), dtype=np.int64))
            for fi in range(n_fragments):
                orig_indices_list.append(asset_gid * 100000 + fi)

            total_fragments += n_fragments
            asset_fragments.append(n_fragments)
            soup_asset_gids.append(asset_gid)

        soup_emb = np.concatenate(soup_emb_list, axis=0)
        soup_trans = np.concatenate(soup_trans_list, axis=0)
        asset_ids = np.concatenate(asset_ids_list, axis=0)
        orig_indices = np.array(orig_indices_list, dtype=np.int64)

        # Scale normalized asset offsets by random factor (uniform range per asset)
        asset_scale_range = getattr(self, 'asset_scale_std')
        if len(asset_scale_range) == 2:
            offset = 0
            for i in range(len(asset_fragments)):
                asset_gid = soup_asset_gids[i]
                n_fragments = asset_fragments[i]
                scale = rng.uniform(asset_scale_range[0], asset_scale_range[1])
                scale = math.pow(10, scale)

                # Scale full 3x3 rotation/scale block + translation (row-major 4x4)
                # Indices: 0-2 (row0), 4-6 (row1), 8-10 (row2) = rotation; 3,7,11 = translation
                for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                    soup_trans[offset:offset + n_fragments, idx] *= scale
                offset += n_fragments

        # Random translation augmentation: per-asset translation in world space
        translation_scale_range = getattr(self, 'translation_scale')
        offset = 0
        sigma = rng.lognormal(translation_scale_range[0], translation_scale_range[1])

        cluster_scale_range = getattr(self, 'cluster_translation_scale')
        sigma2 = rng.lognormal(cluster_scale_range[0], cluster_scale_range[1])

        max_base = 3 + len(asset_fragments) // 10
        cluster_weights = np.array([0.1 + 1 / (2 ** x) for x in range(max_base)], dtype=np.float32)
        cluster_weights /= cluster_weights.sum()
        num_base = int(rng.choice(max_base, p=cluster_weights)) + 1
        base_positions = [rng.randn(3).astype(np.float32) * sigma2 for _ in range(num_base)]

        for i in range(len(asset_fragments)):
            asset_gid = soup_asset_gids[i]
            n_fragments = asset_fragments[i]

            # Pick a base position (50/50) and add per-asset random offset
            chosen = int(rng.choice(num_base))
            t = base_positions[chosen] + rng.randn(3).astype(np.float32) * sigma
            # Translation is at indices 3, 7, 11 of each row (4th column of row-major 4x4)
            soup_trans[offset:offset + n_fragments, 3] += t[0]
            soup_trans[offset:offset + n_fragments, 7] += t[1]
            soup_trans[offset:offset + n_fragments, 11] += t[2]
            offset += n_fragments

        # Random scene scaling (uniform factor for entire soup)
        scene_scale_range = getattr(self, 'scene_scale')
        scene_factor = rng.uniform(scene_scale_range[0], scene_scale_range[1])
        scene_factor = math.pow(10, scene_factor)
        
        # Scale full 3x3 rotation/scale block + translation (row-major 4x4)
        for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            soup_trans[:, idx] *= scene_factor

        # Shuffle soup
        shuffle_idx = rng.permutation(len(soup_emb))
        soup_emb = soup_emb[shuffle_idx]
        soup_trans = soup_trans[shuffle_idx]
        asset_ids = asset_ids[shuffle_idx]
        orig_indices = orig_indices[shuffle_idx]

        # Build N x N ground truth: Y_ij = 1 if same asset (bool, 1/4 memory of float32)
        Y = (asset_ids[:, None] == asset_ids[None, :])

        return {
            'embeddings': torch.from_numpy(soup_emb),
            'transforms': torch.from_numpy(soup_trans),
            'Y': torch.from_numpy(Y),
            'orig_indices': torch.from_numpy(orig_indices),
            'asset_ids': torch.from_numpy(asset_ids),
            'soup_count': k,
            'soup_stats': {
                'num_assets': len(asset_fragments),
                'avg_fragments': sum(asset_fragments) / len(asset_fragments) if asset_fragments else 0,
                'asset_fragments': list(asset_fragments),
                'max_total_fragments': MAX_TOTAL_FRAGMENTS,
            },
        }

    def compute_stats(self, num_samples=None):
        """Compute dataset statistics by iterating samples once.

        Args:
            num_samples: Number of samples to process. None = entire dataset.

        Returns:
            (asset_counts, asset_fragments, budgets) tuples for histogram widget
        """
        asset_counts = []
        asset_fragments = []
        budgets = []
        n = num_samples or len(self)
        for i in range(n):
            sample = self[i]
            stats = sample.get('soup_stats', {})
            if stats:
                asset_counts.append(stats['num_assets'])
                asset_fragments.append(tuple(stats['asset_fragments']))
                budgets.append(stats['max_total_fragments'])
        return asset_counts, asset_fragments, budgets

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
