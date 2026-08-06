"""Scene joining utilities for PeelerDataset.

Combines multiple small scenes into larger joined scenes to fill larger
buckets more strongly. Handles:
- Bounding sphere computation per scene
- Mean scale normalization across scenes
- Greedy spiral packing on XZ plane
"""
import math
import random
import numpy as np

# Direct diagonal indices for 3x3 rotation block in flattened 16-element transform
ROTATION_COLS = (0, 5, 10)
# Columns [3, 7, 11] = translation vector (x, y, z)
TRANSLATION_COLS = (3, 7, 11)


def compute_asset_scale(transform):
    """Compute the scale of an asset from its transform rotation diagonal.

    Args:
        transform: (N, 16) transform array for one asset

    Returns:
        float: mean scale across all fragments
    """
    diag_sum = (
        np.abs(transform[:, 0]).sum()
        + np.abs(transform[:, 5]).sum()
        + np.abs(transform[:, 10]).sum()
    )
    return float(diag_sum / len(transform))


def compute_asset_centroid(transform):
    """Compute the centroid of an asset from its transform translation columns.

    Args:
        transform: (N, 16) transform array for one asset

    Returns:
        np.ndarray: (3,) centroid position
    """
    return np.array(
        [
            transform[:, 3].mean(dtype=np.float64),
            transform[:, 7].mean(dtype=np.float64),
            transform[:, 11].mean(dtype=np.float64),
        ],
        dtype=np.float64,
    )


class SceneJoiner:
    """Orchestrates scene joining: bounding spheres, normalization, packing."""

    def __init__(self, all_transforms, bucket_order, all_embeddings=None,
                 scene_assets=None, scene_is_joined=None, scene_sources=None,
                 scene_bucket=None, bucket_scenes=None, bucket_counts=None,
                 scene_is_sub=None):
        """Initialize scene joiner."""
        self.all_transforms = all_transforms
        self.bucket_order = bucket_order
        self._bucket_index = {bk: i for i, bk in enumerate(bucket_order)}
        self.scene_assets = scene_assets
        self.scene_is_joined = scene_is_joined
        self.scene_sources = scene_sources
        self.scene_bucket = scene_bucket
        self.bucket_scenes = bucket_scenes
        self.bucket_counts = bucket_counts
        self.scene_is_sub = scene_is_sub

        n_assets = len(all_transforms)
        self._asset_scales = np.empty(n_assets, dtype=np.float64)
        self._asset_centroids = np.empty((n_assets, 3), dtype=np.float64)

        for i, t in enumerate(all_transforms):
            inv_len = 1.0 / len(t)
            diag_sum = (
                np.abs(t[:, 0]).sum()
                + np.abs(t[:, 5]).sum()
                + np.abs(t[:, 10]).sum()
            )
            self._asset_scales[i] = diag_sum * inv_len
            self._asset_centroids[i] = (
                t[:, 3].mean(),
                t[:, 7].mean(),
                t[:, 11].mean(),
            )

        if all_embeddings is not None:
            self._asset_frag_counts_list = [len(e) for e in all_embeddings]
        else:
            self._asset_frag_counts_list = [len(t) for t in all_transforms]

        self._asset_frag_counts = np.array(
            self._asset_frag_counts_list, dtype=np.int64
        )

        # Pre-stack trigonometric unit direction vectors (2, Angles) for matrix products
        self._angles_fine = np.arange(0.0, 2 * np.pi + 0.01, 0.15)
        self._dirs_fine = np.stack(
            [np.cos(self._angles_fine), np.sin(self._angles_fine)], axis=0
        )

        self._angles_coarse = np.arange(0.0, 2 * np.pi + 0.01, 0.25)
        self._dirs_coarse = np.stack(
            [np.cos(self._angles_coarse), np.sin(self._angles_coarse)], axis=0
        )

    def get_bucket_rank(self, bucket_key):
        """Return the rank of a bucket (lower = smaller)."""
        return self._bucket_index.get(bucket_key, -1)

    def compute_scene_centroid(self, asset_indices):
        """Compute the centroid of a scene (mean of asset centroids)."""
        return self._asset_centroids[asset_indices].mean(axis=0)

    def compute_scene_bounding_radius(self, asset_indices, centroid=None):
        """Compute the bounding sphere radius of a scene."""
        if len(asset_indices) == 0:
            return 0.0

        centroids = self._asset_centroids[asset_indices]
        if centroid is None:
            centroid = centroids.mean(axis=0)

        diffs = centroids - centroid
        sq_dists = np.einsum('ij,ij->i', diffs, diffs)
        return math.sqrt(float(sq_dists.max()))

    def compute_scene_mean_scale(self, asset_indices):
        """Compute the mean scale of all assets in a scene."""
        if len(asset_indices) == 0:
            return 1.0
        return float(self._asset_scales[asset_indices].mean())

    def pack_scenes(self, sources, radii):
        """Pack circles on XZ plane using greedy spiral search."""
        n = len(radii)
        if n == 0:
            return []

        positions = np.zeros((n, 2), dtype=np.float64)
        radii_arr = np.asarray(radii, dtype=np.float64)

        pos_sq_arr = np.zeros(n, dtype=np.float64)
        dists_plus_radii = np.zeros(n, dtype=np.float64)
        dists_plus_radii[0] = radii_arr[0]

        for i in range(1, n):
            radius = radii_arr[i]
            curr_pos = positions[:i]
            curr_radii = radii_arr[:i]
            pos_sq = pos_sq_arr[:i]

            max_dist = np.max(dists_plus_radii[:i]) + radius
            search_max = max(max_dist * 2.0, radius * 4.0)

            dirs = self._dirs_coarse if i > 10 else self._dirs_fine
            min_gaps_sq = (curr_radii + radius + 1e-6) ** 2  # Shape: (i,)

            proj = curr_pos @ dirs  # Shape: (i, Angles)
            const_term = (pos_sq - min_gaps_sq)[:, None]  # Shape: (i, 1)

            # Quadratic overlap interval discriminant: R^2 - 2 R proj + const <= 0
            disc = proj ** 2 - const_term  # Shape: (i, Angles)

            # For disc <= 0, sqrt_disc = 0.0 => r1 = r2 = proj.
            # (R > proj) & (R < proj) is strictly False, so no masking needed.
            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            r1 = proj - sqrt_disc
            r2 = proj + sqrt_disc

            num_angles = dirs.shape[1]
            R = np.full(num_angles, radius * 2.0, dtype=np.float64)

            # Iteratively advance R past any overlapping interval (at most i steps)
            for _ in range(i):
                in_range = (R > r1) & (R < r2)
                if not in_range.any():
                    break
                R = np.maximum(R, np.where(in_range, r2, -np.inf).max(axis=0))

            # Quantize R to 0.5 grid steps matching original discrete search
            k = np.ceil((R - radius * 2.0) * 2.0)
            R_grid = radius * 2.0 + np.maximum(k, 0.0) * 0.5

            best_a = int(np.argmin(R_grid))
            best_r = R_grid[best_a]

            if best_r <= search_max + 1e-9:
                positions[i, 0] = best_r * dirs[0, best_a]
                positions[i, 1] = best_r * dirs[1, best_a]
            else:
                positions[i, 0] = np.max(curr_pos[:, 0]) + radius * 2.0
                positions[i, 1] = 0.0

            p0, p1 = positions[i, 0], positions[i, 1]
            psq = p0 * p0 + p1 * p1
            pos_sq_arr[i] = psq
            dists_plus_radii[i] = math.sqrt(psq) + radii_arr[i]

        return [tuple(p) for p in positions]

    def compute_join_transforms(self, source_scene_indices, scene_assets):
        """Compute normalization and packing transforms for a group of source scenes."""
        transforms_map = {}

        for scene_idx in source_scene_indices:
            asset_indices = scene_assets[scene_idx]
            if len(asset_indices) == 0:
                transforms_map[scene_idx] = {
                    'centroid': np.zeros(3, dtype=np.float64),
                    'mean_scale': 1.0,
                    'bounding_radius': 0.0,
                    'asset_norm_factors': {},
                }
                continue

            centroids = self._asset_centroids[asset_indices]
            scales = self._asset_scales[asset_indices]

            mean_scale = float(scales.mean())
            centroid = centroids.mean(axis=0)

            diffs = centroids - centroid
            sq_dists = np.einsum('ij,ij->i', diffs, diffs)
            bounding_radius = math.sqrt(float(sq_dists.max()))

            inv_scale = (1.0 / mean_scale) if mean_scale > 0 else 1.0
            asset_norm_factors = dict(zip(asset_indices, (scales * inv_scale).tolist()))

            transforms_map[scene_idx] = {
                'centroid': centroid,
                'mean_scale': mean_scale,
                'bounding_radius': bounding_radius,
                'asset_norm_factors': asset_norm_factors,
            }

        return transforms_map

    def select_scenes_for_bucket(self, available_scenes, scene_assets,
                                   bucket_lower, bucket_upper):
        """Select and group scenes for joining."""
        groups = []
        remaining = list(available_scenes)
        random.shuffle(remaining)

        frag_counts = self._asset_frag_counts_list
        scene_frag_counts = {
            s: sum([frag_counts[a] for a in scene_assets[s]])
            for s in set(remaining)
        }
        counts = [scene_frag_counts[s] for s in remaining]

        i = 0
        n_rem = len(remaining)
        while i < n_rem:
            total_frags = counts[i]

            if total_frags >= bucket_upper:
                i += 1
                continue

            group = [remaining[i]]
            j = i + 1

            while j < n_rem:
                next_frags = counts[j]
                if total_frags + next_frags < bucket_upper:
                    group.append(remaining[j])
                    total_frags += next_frags
                    j += 1
                else:
                    break

            if total_frags >= bucket_lower:
                groups.append(group)

            i = j if len(group) > 1 else i + 1

        return groups

    def run(self, max_fragments):
        """Run the full join pipeline: iterate buckets and join scenes."""
        from .training.validate import FRAGMENT_RANGES
        from .dataset import MAX_BUCKET_SIZE

        next_idx = max(self.scene_assets.keys()) + 1

        for bucket_idx, (bucket_range, bucket_key) in enumerate(FRAGMENT_RANGES):
            bucket_lower, bucket_upper = bucket_range
            if bucket_key == '300+':
                bucket_upper = max_fragments

            available = []
            for smaller_idx in range(bucket_idx):
                _, smaller_key = FRAGMENT_RANGES[smaller_idx]
                for scene_idx in self.bucket_scenes.get(smaller_key, []):
                    if scene_idx not in self.scene_is_sub and scene_idx not in self.scene_is_joined:
                        available.append(scene_idx)

            if len(available) < 2:
                continue

            groups = self.select_scenes_for_bucket(
                available, self.scene_assets,
                bucket_lower, bucket_upper,
            )

            for group in groups:
                if len(group) < 2:
                    continue
                if self.bucket_counts[bucket_key] >= MAX_BUCKET_SIZE:
                    continue

                join_transforms = self.compute_join_transforms(group, self.scene_assets)
                sources = list(join_transforms.keys())
                radii = [join_transforms[s]['bounding_radius'] for s in sources]
                positions = self.pack_scenes(sources, radii)

                pos_map = dict(zip(sources, positions))

                joined_assets = []
                for src in group:
                    joined_assets.extend(self.scene_assets[src])

                self.scene_assets[next_idx] = joined_assets
                self.scene_is_joined[next_idx] = True
                self.scene_sources[next_idx] = group

                # Apply transforms directly to all_transforms (scale normalize + pack)
                for src in group:
                    t = join_transforms[src]
                    mean_scale = t['mean_scale']
                    inv_scale = (1.0 / mean_scale) if mean_scale > 0 else 1.0
                    pack_x = pos_map[src][0]
                    pack_z = pos_map[src][1]

                    for asset_gid in self.scene_assets[src]:
                        norm_factor = t['asset_norm_factors'][asset_gid]
                        if not (0.01 < norm_factor < 100.0):
                            continue
                        trans = self.all_transforms[asset_gid]
                        trans[:, ROTATION_COLS] *= norm_factor
                        trans[:, TRANSLATION_COLS] *= norm_factor
                        trans[:, 3] += pack_x
                        trans[:, 11] += pack_z

                self.scene_bucket[next_idx] = bucket_key
                self.bucket_scenes[bucket_key].append(next_idx)
                self.bucket_counts[bucket_key] += 1

                next_idx += 1