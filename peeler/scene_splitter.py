"""Scene splitting utilities for PeelerDataset.

Splits large scenes into sub-scenes for smaller buckets using greedy
accumulation of consecutive assets in Morton order.
"""
import numpy as np


class SceneSplitter:
    """Splits scenes into sub-scenes for smaller buckets."""

    def __init__(self, scene_assets, scene_is_sub, scene_parent, scene_bucket,
                 bucket_scenes, bucket_counts, all_embeddings, max_fragments, mode):
        self.scene_assets = scene_assets
        self.scene_is_sub = scene_is_sub
        self.scene_parent = scene_parent
        self.scene_bucket = scene_bucket
        self.bucket_scenes = bucket_scenes
        self.bucket_counts = bucket_counts
        self.all_embeddings = all_embeddings
        self.max_fragments = max_fragments
        self.mode = mode

    def run(self):
        """Run the full split pipeline: iterate scenes and smaller buckets."""
        from .training.validate import FRAGMENT_RANGES
        from .dataset import _frag_to_range_key, MAX_BUCKET_SIZE

        next_idx = max(self.scene_assets.keys()) + 1 if self.scene_assets else 0

        for scene_idx in list(self.scene_assets.keys()):
            assets = self.scene_assets[scene_idx]
            frag_counts = [len(self.all_embeddings[a]) for a in assets]
            total_frags = sum(frag_counts)
            scene_bucket = _frag_to_range_key(total_frags)

            for (low, high), bucket_key in FRAGMENT_RANGES:
                if bucket_key == '1' or bucket_key == scene_bucket:
                    continue

                if self.bucket_counts[bucket_key] >= MAX_BUCKET_SIZE:
                    continue

                sub_asset_lists = self._split_scene(assets, frag_counts, low, high)
                for asset_indices in sub_asset_lists:
                    self.scene_assets[next_idx] = asset_indices
                    self.scene_is_sub[next_idx] = True
                    self.scene_parent[next_idx] = scene_idx
                    self.scene_bucket[next_idx] = bucket_key
                    self.bucket_scenes[bucket_key].append(next_idx)
                    self.bucket_counts[bucket_key] += 1
                    next_idx += 1

    def _split_scene(self, assets, frag_counts, bucket_lower, bucket_upper):
        """Split a scene into sub-scenes for a given bucket range.

        Greedily accumulates consecutive assets starting from each position.
        Returns list of asset index lists, each fitting within [bucket_lower, bucket_upper).
        """
        n = len(assets)
        sub_scenes = []
        i = 0

        while i < n:
            fi = frag_counts[i]

            if fi >= bucket_upper:
                i += 1
                continue

            total = fi
            j = i + 1
            while j < n and total + frag_counts[j] < bucket_upper:
                total += frag_counts[j]
                j += 1

            if total >= bucket_lower and total < bucket_upper:
                sub_scenes.append([assets[k] for k in range(i, j)])

            i = j

        return sub_scenes
