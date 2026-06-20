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
        max_assets_per_soup: int, maximum number of assets to mix per soup (default 10)
        min_assets_per_soup: int, minimum number of assets per soup (default 2)
        seed: int, random seed for reproducibility
    """

    def __init__(
        self,
        all_embeddings,
        all_transforms,
        max_assets_per_soup,
        min_assets_per_soup,
        max_asset_fragments,
        seed,
        translation_scale,
        asset_scale_std,
        scene_scale
    ):
        self.all_embeddings = all_embeddings  # list of (N_i, 256)
        self.all_transforms = all_transforms  # list of (N_i, 16)
        self.max_assets_per_soup = max_assets_per_soup
        self.min_assets_per_soup = min_assets_per_soup
        self.max_asset_fragments = max_asset_fragments
        self.translation_scale = translation_scale
        self.asset_scale_std = asset_scale_std
        self.scene_scale = scene_scale
        self.seed = seed
        self._epoch = 0

        # Subsample oversized assets to cap per-asset fragment count
        if max_asset_fragments is not None:
            rng = np.random.RandomState(seed)
            subsampled_emb = []
            subsampled_trans = []
            for emb, trans in zip(all_embeddings, all_transforms):
                n = len(emb)
                if n > max_asset_fragments:
                    indices = rng.choice(n, size=max_asset_fragments, replace=False)
                    indices = np.sort(indices)
                    subsampled_emb.append(emb[indices])
                    subsampled_trans.append(trans[indices])
                else:
                    subsampled_emb.append(emb)
                    subsampled_trans.append(trans)
            self.all_embeddings = subsampled_emb
            self.all_transforms = subsampled_trans

        self.n_assets = len(self.all_embeddings)

    def set_epoch(self, epoch: int):
        """Set the epoch for randomization."""
        self._epoch = epoch

    def __len__(self):
        return self.n_assets
    
    def get_exponential_weights(self, min_val, max_val, decay):
        x = np.arange(min_val, max_val + 1)
        # Compute raw exponential weights
        raw_weights = np.exp(-decay * x)
        # Normalize to sum to exactly 1.0
        return raw_weights / raw_weights.sum()
    
    def __getitem__(self, idx):
        # Create fresh RNG seeded by epoch + idx for full randomization
        rng = np.random.RandomState(self.seed + idx + self._epoch * 100000)

        # Sample K assets with random size in [min, max]
        upper = min(self.max_assets_per_soup, self.n_assets)
        lower = min(self.min_assets_per_soup, upper)
        
        # 1. Generate the possible choices of K (e.g., [1, 2, 3, ..., 12])
        choices = np.arange(lower, upper + 1)
        
        # 2. Get the exponential decay probabilities specifically for this valid range [1]
        probs = self.get_exponential_weights(lower, upper, 0.3)
        
        # 3. Sample a single integer K using the decay probabilities [1]
        # rng.choice will select exactly 1 integer according to your skewed weights
        k = int(rng.choice(choices, p=probs))  # Cast to standard python int for safety
        
        # Now k is a clean, single integer (e.g., 1, 2, or 3) weighted toward the low end!
        asset_indices = list(rng.choice(self.n_assets, size=k, replace=False))

        # Combine all fragments from selected assets
        soup_emb_list = []
        soup_trans_list = []
        asset_ids_list = []
        orig_indices_list = []

        for aid, asset_gid in enumerate(asset_indices):
            emb = self.all_embeddings[asset_gid]
            trans = self.all_transforms[asset_gid]
            soup_emb_list.append(emb)
            soup_trans_list.append(trans)
            asset_ids_list.append(np.full(len(emb), aid, dtype=np.int64))
            # Encode (asset_gid, frag_idx) uniformly
            for fi in range(len(emb)):
                orig_indices_list.append(asset_gid * 100000 + fi)

        soup_emb = np.concatenate(soup_emb_list, axis=0)
        soup_trans = np.concatenate(soup_trans_list, axis=0)
        asset_ids = np.concatenate(asset_ids_list, axis=0)
        orig_indices = np.array(orig_indices_list, dtype=np.int64)

        # Scale normalized asset offsets by random factor (uniform range per asset)
        asset_scale_range = getattr(self, 'asset_scale_std')
        if len(asset_scale_range) == 2:
            offset = 0
            for asset_gid in asset_indices:
                n_fragments = len(self.all_transforms[asset_gid])
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
        for asset_gid in asset_indices:
            n_fragments = len(self.all_transforms[asset_gid])
            
            t = rng.randn(3).astype(np.float32) * sigma
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
        }

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
        }
