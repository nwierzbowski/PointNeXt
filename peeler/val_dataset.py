"""PeelerValidationDataset - Pre-made soup validation for peeler model.

Each TBO file is treated as a complete soup. No random mixing, no data
augmentation. Assets within a TBO are concatenated and the Y matrix
encodes same-asset membership across all assets in the soup.

Reuses PeelerDataset.collate_fn for batch padding.
"""
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset

from openpoints.dataset.build import DATASETS


@DATASETS.register_module()
class PeelerValidationDataset(Dataset):
    """Validation dataset where each TBO file = one soup.

    No random sampling, no augmentation. Each soup contains all assets
    from one TBO file, concatenated together. Y encodes same-asset
    membership for all fragments in the soup.

    Args:
        all_embeddings: list of numpy arrays (N_i, 256), one per asset
        all_transforms: list of numpy arrays (N_i, 16), one per asset
        asset_to_file: list mapping asset_idx -> file_idx (groups assets by TBO file)
        max_fragments: unused (no truncation, collate handles padding)
        seed: unused (no randomization)
    """

    def __init__(
        self,
        all_embeddings,
        all_transforms,
        asset_to_file,
        max_fragments,
        seed,
    ):
        self.all_embeddings = all_embeddings
        self.all_transforms = all_transforms
        self.max_fragments = max_fragments
        self.seed = seed

        # Group asset indices by file
        self._file_assets = defaultdict(list)
        for asset_idx, file_idx in enumerate(asset_to_file):
            self._file_assets[file_idx].append(asset_idx)

        # Sort file indices for deterministic ordering
        self._file_indices = sorted(self._file_assets.keys())

    def __len__(self):
        return len(self._file_indices)

    def __getitem__(self, idx):
        file_idx = self._file_indices[idx]
        asset_indices = self._file_assets[file_idx]

        # Concatenate all assets in this TBO into one soup
        soup_emb_list = []
        soup_trans_list = []
        asset_ids_list = []

        for local_asset_id, asset_idx in enumerate(asset_indices):
            emb = self.all_embeddings[asset_idx]
            trans = self.all_transforms[asset_idx]
            n = len(emb)

            soup_emb_list.append(emb)
            soup_trans_list.append(trans)
            asset_ids_list.append(np.full(n, local_asset_id, dtype=np.int64))

        soup_emb = np.concatenate(soup_emb_list, axis=0)
        soup_trans = np.concatenate(soup_trans_list, axis=0)
        asset_ids_np = np.concatenate(asset_ids_list, axis=0)
        asset_ids = torch.from_numpy(asset_ids_np)

        n = len(soup_emb)

        # Y matrix: Y[i,j] = 1 if fragments i,j belong to same asset
        Y = (asset_ids[:, None] == asset_ids[None, :])

        num_assets = len(asset_indices)
        asset_fragments = [len(self.all_embeddings[i]) for i in asset_indices]

        return {
            'embeddings': torch.from_numpy(soup_emb),
            'transforms': torch.from_numpy(soup_trans),
            'Y': Y,
            'orig_indices': torch.arange(n, dtype=torch.int64),
            'asset_ids': asset_ids,
            'soup_count': num_assets,
            'soup_stats': {
                'num_assets': num_assets,
                'avg_fragments': sum(asset_fragments) / num_assets if num_assets else 0,
                'asset_fragments': asset_fragments,
                'actual_n': n,
            },
        }

    @staticmethod
    def collate_fn(datas):
        from peeler.dataset import PeelerDataset
        return PeelerDataset.collate_fn(datas)
