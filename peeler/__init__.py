"""Peeler - Adaptive Object Peeler for 3D fragment grouping.

Submodules:
    model: PurelyRelationalPeeler
    dataset: PeelerDataset
    loss: AlignedPullPushPeelerLoss
    training: setup and train pipeline
"""
from peeler.model import PurelyRelationalPeeler
from peeler.dataset import PeelerDataset
from peeler.loss import AlignedPullPushPeelerLoss

__all__ = [
    'PurelyRelationalPeeler',
    'PeelerDataset',
    'AlignedPullPushPeelerLoss',
]
