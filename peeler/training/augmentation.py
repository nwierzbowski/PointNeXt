"""Random augmentation utilities for PeelerDataset soup generation.

Applies stochastic transformations to fragment transform arrays to
increase training diversity. Designed as a standalone module so real
transform sources can be swapped in alongside or instead of random.
"""
import math

import numpy as np


class AugmentationEngine:
    """Stateless random augmentation for peeler soup transforms.

    All methods operate in-place on the ``soup_trans`` array.

    Args:
        translation_noise_sigma: std dev of per-fragment Gaussian translation noise
        scaling_noise_sigma: std dev of per-fragment Gaussian scaling noise
        per_asset_rotation: whether to apply uniform SO(3) rotation per asset
        asset_scale_std: tuple (low, high) for per-asset scale variation
        translation_scale: tuple (mean, std) for lognormal cluster translation
        cluster_translation_scale: tuple (low, high) for cluster spacing range
        scene_scale: tuple (low, high) for uniform scene scaling
    """

    def __init__(
        self,
        translation_noise_sigma: float = 0.0,
        scaling_noise_sigma: float = 0.0,
        per_asset_rotation: bool = True,
        asset_scale_std = (0.0, 1.0),
        translation_scale = (0.0, 0.5),
        cluster_translation_scale = (1.0, 5.0),
        scene_scale = (-1.0, 1.0),
    ):
        self.translation_noise_sigma = translation_noise_sigma
        self.scaling_noise_sigma = scaling_noise_sigma
        self.per_asset_rotation = per_asset_rotation
        self.asset_scale_std = asset_scale_std
        self.translation_scale = translation_scale
        self.cluster_translation_scale = cluster_translation_scale
        self.scene_scale = scene_scale

    def apply_full_random(self, soup_trans, asset_fragments, soup_asset_gids, rng, k, scene_transforms):
        """Apply the full random augmentation pipeline in-place.

        Args:
            soup_trans: (N, 16) transform array, modified in-place
            asset_fragments: list of fragment counts per asset in soup
            soup_asset_gids: list of asset global IDs
            rng: numpy RandomState instance
            k: number of assets in the soup
            scene_transforms: list of numpy arrays, one per scene (from transforms TBO), each (M_i, 16) 4x4 matrices
        """
        self._apply_translation_noise(soup_trans, asset_fragments, rng)
        self._apply_scaling_noise(soup_trans, asset_fragments, rng)
        
        # 0-------------1
        # synth-----scene
        synth_prob = 0.8
        
        # 50/50 choice between scene transforms and cluster translation
        if rng.random() > synth_prob:
            self.apply_scene_transforms(soup_trans, asset_fragments, soup_asset_gids, rng, k, scene_transforms)
        else:
            if self.per_asset_rotation:
                self._apply_per_asset_rotation(soup_trans, asset_fragments, rng)
            self._apply_per_asset_scaling(soup_trans, asset_fragments, rng)
            self._apply_cluster_translation(soup_trans, asset_fragments, soup_asset_gids, rng, k)

    def apply_scene_transforms(self, soup_trans, asset_fragments, soup_asset_gids, rng, k, scene_transforms):
        """Apply real scene transforms to soup assets in-place.

        Picks one random scene from the transforms TBO and applies its transforms
        sequentially to soup assets (1 per asset). When the scene runs out,
        picks a new random scene and continues.

        Args:
            soup_trans: (N, 16) transform array, modified in-place
            asset_fragments: list of fragment counts per asset in soup
            soup_asset_gids: list of asset global IDs
            rng: numpy RandomState instance
            k: number of assets in the soup
            scene_transforms: list of numpy arrays, one per scene (from transforms TBO), each (M_i, 16) 4x4 matrices
        """
        num_assets = len(asset_fragments)
        scales = rng.randn(num_assets) * self.asset_scale_std[1] + 1
        translations = (rng.randn(num_assets, 3) * self.translation_scale[1] + 1).astype(np.float32)

        offset = 0
        scene_idx = rng.randint(len(scene_transforms))
        transforms = scene_transforms[scene_idx]
        perm = rng.permutation(len(transforms))
        t_idx = 0

        for i in range(num_assets):
            n = asset_fragments[i]

            # When current scene runs out, pick a new random scene
            if t_idx >= len(transforms):
                scene_idx = rng.randint(len(scene_transforms))
                transforms = scene_transforms[scene_idx]
                perm = rng.permutation(len(transforms))
                t_idx = 0

            matrix = transforms[perm[t_idx]].reshape(4, 4).astype(np.float64)
            R = matrix[:3, :3]
            t = matrix[:3, 3]

            # Apply rotation to rotation/scale block, then scale by asset's scale factor
            rot_blocks = soup_trans[offset:offset + n, :12].reshape(n, 3, 4)[:, :, :3].astype(np.float64)
            soup_trans[offset:offset + n, [0, 1, 2, 4, 5, 6, 8, 9, 10]] = ((R @ rot_blocks) * scales[i]).astype(np.float32).reshape(n, 9)

            # Apply rotation to translation, then add scene translation and per-asset translation offset
            asset_translations = soup_trans[offset:offset + n, [3, 7, 11]].astype(np.float64)
            soup_trans[offset:offset + n, [3, 7, 11]] = (asset_translations @ R.T + t * translations[i]).astype(np.float32)

            t_idx += 1
            offset += n

    # ------------------------------------------------------------------
    # Individual augmentation steps
    # ------------------------------------------------------------------

    def _apply_translation_noise(self, soup_trans, asset_fragments, rng):
        if self.translation_noise_sigma > 0:
            offset = 0
            for i in range(len(asset_fragments)):
                n = asset_fragments[i]
                noise = rng.randn(n, 3).astype(np.float32) * self.translation_noise_sigma
                soup_trans[offset:offset + n, 3] += noise[:, 0]
                soup_trans[offset:offset + n, 7] += noise[:, 1]
                soup_trans[offset:offset + n, 11] += noise[:, 2]
                offset += n

    def _apply_scaling_noise(self, soup_trans, asset_fragments, rng):
        if self.scaling_noise_sigma > 0:
            offset = 0
            for i in range(len(asset_fragments)):
                n = asset_fragments[i]
                # Generate a SINGLE isotropic scale factor per fragment
                noise = rng.randn(n, 1).astype(np.float32) * self.scaling_noise_sigma + 1.0
                
                # Multiply all 9 components of the 3x3 matrix by the same scalar
                soup_trans[offset:offset + n, [0, 1, 2, 4, 5, 6, 8, 9, 10]] *= noise
                offset += n

    def _apply_per_asset_rotation(self, soup_trans, asset_fragments, rng):
        offset = 0
        for i in range(len(asset_fragments)):
            n = asset_fragments[i]
            R = self._uniform_rotation(rng)
            translations = soup_trans[offset:offset + n, [3, 7, 11]].astype(np.float64)
            soup_trans[offset:offset + n, [3, 7, 11]] = translations @ R.T.astype(np.float32)
            rot_blocks = soup_trans[offset:offset + n, :12].reshape(n, 3, 4)[:, :, :3].astype(np.float64)
            soup_trans[offset:offset + n, [0, 1, 2, 4, 5, 6, 8, 9, 10]] = (R @ rot_blocks).astype(np.float32).reshape(n, 9)
            offset += n

    def _apply_per_asset_scaling(self, soup_trans, asset_fragments, rng):
        offset = 0
        for i in range(len(asset_fragments)):
            n_fragments = asset_fragments[i]
            scale = min(rng.exponential(scale=1.27), 2)
            scale = math.pow(10, scale)
            for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                soup_trans[offset:offset + n_fragments, idx] *= scale
            offset += n_fragments

    def _apply_cluster_translation(self, soup_trans, asset_fragments, soup_asset_gids, rng, k):
        
        max_base = 3 + k // 4
        cluster_weights = np.array([0.05 + 2 ** -x for x in range(max_base)], dtype=np.float32)
        cluster_weights /= cluster_weights.sum()
        num_base = int(rng.choice(max_base, p=cluster_weights)) + 1

        base_axis_scales = rng.dirichlet(np.full(3, 0.5)).astype(np.float32)
        c_low, c_high = self.cluster_translation_scale
        base_positions = rng.uniform(c_low, c_high, size=(num_base, 3)).astype(np.float32) * base_axis_scales

        R_base = self._uniform_rotation(rng)
        base_positions = base_positions @ R_base.T
        intra_rotations = [self._uniform_rotation(rng) for _ in range(num_base)]

        sigma = rng.lognormal(0, 0.34)
        alpha_cluster_prob = np.exp(rng.uniform(-1.0, 2.0))
        cluster_probs = rng.dirichlet(np.full(num_base, alpha_cluster_prob))
        axis_scales_list = rng.dirichlet(np.full(3, 0.5), size=num_base).astype(np.float32)
        offset = 0
        for i in range(len(asset_fragments)):
            n = asset_fragments[i]
            chosen = int(rng.choice(num_base, p=cluster_probs))
            intra_offset = (rng.randn(3) * axis_scales_list[chosen]).astype(np.float32) * sigma
            intra_offset = intra_rotations[chosen] @ intra_offset
            t = base_positions[chosen] + intra_offset
            soup_trans[offset:offset + n, 3] += t[0]
            soup_trans[offset:offset + n, 7] += t[1]
            soup_trans[offset:offset + n, 11] += t[2]
            offset += n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _uniform_rotation(rng):
        """Generate a uniform random 3x3 rotation matrix (Marsaglia method)."""
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        theta = rng.uniform(0, np.pi)
        c = np.cos(theta)
        s = np.sin(theta)
        t = 1 - c
        return np.array([
            [t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
            [t * axis[1] * axis[0] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]],
            [t * axis[2] * axis[0] - s * axis[1], t * axis[2] * axis[1] + s * axis[0], t * axis[2] * axis[2] + c],
        ], dtype=np.float32)
