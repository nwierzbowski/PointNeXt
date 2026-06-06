"""Peeler - Adaptive Object Peeler for 3D fragment grouping.

Submodules:
    model: PeelerBackbone, PeelerLoop, Peeler
    dataset: PeelerDataset
    loss: PeelerLoss
    training: setup and train pipeline
"""
from peeler.model import PeelerBackbone, PeelerLoop, Peeler
from peeler.dataset import PeelerDataset
from peeler.loss import PeelerLoss

__all__ = [
    'PeelerBackbone',
    'PeelerLoop',
    'Peeler',
    'PeelerDataset',
    'PeelerLoss',
]
