"""Peeler - Adaptive Object Peeler for 3D fragment grouping.

Submodules:
    model: Peeler
    dataset: PeelerDataset
    loss: AlignedPullPushPeelerLoss
    training: setup and train pipeline
"""
from peeler.model import Peeler
from peeler.dataset import PeelerDataset
from peeler.loss import AlignedPullPushPeelerLoss

__all__ = [
    'Peeler',
    'PeelerDataset',
    'AlignedPullPushPeelerLoss',
]
