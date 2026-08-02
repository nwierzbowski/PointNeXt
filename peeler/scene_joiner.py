"""Scene joining utilities for PeelerDataset.

Combines multiple small scenes into larger joined scenes to fill larger
buckets more strongly. Handles:
- Bounding sphere computation per scene
- Mean scale normalization across scenes
- Greedy spiral packing on XZ plane

Usage:
    joiner = SceneJoiner(all_transforms, bucket_order)
    groups = joiner.select_scenes_for_bucket(available, scene_assets,
                                              all_embeddings, bucket_lower, bucket_upper)
    transforms = joiner.compute_join_transforms(group, scene_assets)
    positions = joiner.pack_scenes(group, radii)
"""
import numpy as np


# Asset scale is derived from the rotation block diagonal
# Columns [0,1,2,4,5,6,8,9,10] = 3x3 rotation matrix (row-major)
ROTATION_COLS = [0, 1, 2, 4, 5, 6, 8, 9, 10]
# Columns [3,7,11] = translation vector (x, y, z)
TRANSLATION_COLS = [3, 7, 11]


def compute_asset_scale(transform):
    """Compute the scale of an asset from its transform rotation block.

    Args:
        transform: (N, 16) transform array for one asset

    Returns:
        float: mean scale across all fragments (mean of rotation diagonal abs values)
    """
    rot = transform[:, ROTATION_COLS].reshape(-1, 3, 3)
    diag = np.abs(rot[:, 0, 0]) + np.abs(rot[:, 1, 1]) + np.abs(rot[:, 2, 2])
    return float(np.mean(diag))


def compute_asset_centroid(transform):
    """Compute the centroid of an asset from its transform translation columns.

    Args:
        transform: (N, 16) transform array for one asset

    Returns:
        np.ndarray: (3,) centroid position
    """
    return np.mean(transform[:, TRANSLATION_COLS], axis=0)


class SceneJoiner:
    """Orchestrates scene joining: bounding spheres, normalization, packing."""

    def __init__(self, all_transforms, bucket_order):
        """Initialize scene joiner.

        Args:
            all_transforms: list of numpy arrays, one per asset, shape (N_i, 16)
            bucket_order: list of bucket keys in ascending order (e.g. CURRICULUM_BUCKETS)
        """
        self.all_transforms = all_transforms
        self.bucket_order = bucket_order
        self._bucket_index = {bk: i for i, bk in enumerate(bucket_order)}

    def get_bucket_rank(self, bucket_key):
        """Return the rank of a bucket (lower = smaller)."""
        return self._bucket_index.get(bucket_key, -1)

    def compute_scene_centroid(self, asset_indices):
        """Compute the centroid of a scene (mean of asset centroids).

        Args:
            asset_indices: list of asset gids

        Returns:
            np.ndarray: (3,) centroid position
        """
        centroids = [compute_asset_centroid(self.all_transforms[gid]) for gid in asset_indices]
        return np.mean(centroids, axis=0)

    def compute_scene_bounding_radius(self, asset_indices, centroid=None):
        """Compute the bounding sphere radius of a scene.

        Distance from centroid to farthest asset centroid.

        Args:
            asset_indices: list of asset gids
            centroid: optional pre-computed centroid (3,)

        Returns:
            float: bounding sphere radius
        """
        if centroid is None:
            centroid = self.compute_scene_centroid(asset_indices)
        max_dist = 0.0
        for gid in asset_indices:
            asset_centroid = compute_asset_centroid(self.all_transforms[gid])
            dist = np.linalg.norm(asset_centroid - centroid)
            if dist > max_dist:
                max_dist = dist
        return float(max_dist)

    def compute_scene_mean_scale(self, asset_indices):
        """Compute the mean scale of all assets in a scene.

        Args:
            asset_indices: list of asset gids

        Returns:
            float: mean scale
        """
        scales = [compute_asset_scale(self.all_transforms[gid]) for gid in asset_indices]
        return float(np.mean(scales)) if scales else 1.0

    def pack_scenes(self, sources, radii):
        """Pack circles on XZ plane using greedy spiral search.

        Places circles sequentially. First at origin, then searches outward
        along rays at discrete angles until finding a valid non-overlapping position.

        Args:
            sources: list of source scene indices (in packing order)
            radii: list of bounding sphere radii (same order as sources)

        Returns:
            list of (x, z) tuples for each source scene
        """
        positions = []

        for i, radius in enumerate(radii):
            if i == 0:
                positions.append((0.0, 0.0))
                continue

            # Compute search boundary from existing placements
            max_dist = 0.0
            for j, (px, pz) in enumerate(positions):
                d = np.sqrt(px ** 2 + pz ** 2)
                min_dist = d + radii[j] + radius
                if min_dist > max_dist:
                    max_dist = min_dist

            search_max = max(max_dist * 2.0, radius * 4.0)
            angle_step = 0.15  # radians (~8.6 degrees)

            placed = False
            for dist in np.arange(radius * 2.0, search_max + 0.01, 0.5):
                for angle in np.arange(0.0, 2 * np.pi + 0.01, angle_step):
                    x = dist * np.cos(angle)
                    z = dist * np.sin(angle)

                    valid = True
                    for j, (px, pz) in enumerate(positions):
                        dx = x - px
                        dz = z - pz
                        if np.sqrt(dx * dx + dz * dz) < radii[j] + radius + 1e-6:
                            valid = False
                            break

                    if valid:
                        positions.append((float(x), float(z)))
                        placed = True
                        break

                if placed:
                    break

            if not placed:
                x = max((p[0] for p in positions), default=0.0) + radius * 2.0
                positions.append((x, 0.0))

        return positions

    def compute_join_transforms(self, source_scene_indices, scene_assets):
        """Compute normalization and packing transforms for a group of source scenes.

        Args:
            source_scene_indices: list of scene indices to join
            scene_assets: dict scene_idx -> list of asset gids

        Returns:
            dict mapping source_scene_idx -> {
                'centroid': np.ndarray (3,) scene centroid in original coordinates,
                'mean_scale': float,
                'bounding_radius': float,
                'asset_norm_factors': dict mapping asset_gid -> float (asset_scale / mean_scale)
            }
        """
        transforms_map = {}

        for scene_idx in source_scene_indices:
            asset_indices = scene_assets[scene_idx]
            mean_scale = self.compute_scene_mean_scale(asset_indices)
            centroid = self.compute_scene_centroid(asset_indices)
            bounding_radius = self.compute_scene_bounding_radius(asset_indices, centroid)

            # Per-asset normalization factors
            asset_norm_factors = {}
            for gid in asset_indices:
                asset_scale = compute_asset_scale(self.all_transforms[gid])
                asset_norm_factors[gid] = asset_scale / mean_scale if mean_scale > 0 else 1.0

            transforms_map[scene_idx] = {
                'centroid': centroid,
                'mean_scale': mean_scale,
                'bounding_radius': bounding_radius,
                'asset_norm_factors': asset_norm_factors,
            }

        return transforms_map

    def select_scenes_for_bucket(self, available_scenes, scene_assets,
                                   all_embeddings, bucket_lower, bucket_upper):
        """Select and group scenes for joining.

        Randomly permutes available scenes, then greedily accumulates consecutive
        scenes until fragment count fits within [bucket_lower, bucket_upper).

        Args:
            available_scenes: list of scene indices available for joining
            scene_assets: dict scene_idx -> list of asset gids
            all_embeddings: list of numpy arrays (N_i, 256)
            bucket_lower: lower bound of target bucket fragment range (inclusive)
            bucket_upper: upper bound of target bucket fragment range (exclusive)

        Returns:
            list of lists of scene indices (each inner list is a group to join)
        """
        groups = []
        remaining = list(available_scenes)
        np.random.shuffle(remaining)

        i = 0
        while i < len(remaining):
            scene_idx = remaining[i]
            asset_indices = scene_assets[scene_idx]
            total_frags = sum(len(all_embeddings[a]) for a in asset_indices)

            if total_frags >= bucket_upper:
                i += 1
                continue

            # Greedily accumulate consecutive scenes
            group = [scene_idx]
            j = i + 1

            while j < len(remaining):
                next_scene = remaining[j]
                next_assets = scene_assets[next_scene]
                next_frags = sum(len(all_embeddings[a]) for a in next_assets)

                if total_frags + next_frags < bucket_upper:
                    group.append(next_scene)
                    total_frags += next_frags
                    j += 1
                else:
                    break

            # Only create a joined scene if it fits the bucket range
            if total_frags >= bucket_lower:
                groups.append(group)

            i = j if len(group) > 1 else i + 1

        return groups
