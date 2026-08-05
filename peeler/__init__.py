"""Peeler - Purely relational sparse architecture with Top-K sigmoid + focal BCE.

Submodules:
    model: PurelyRelationalPeeler
    dataset: PeelerDataset
    loss: SparseFocalBCELoss
"""
from peeler.model import PurelyRelationalPeeler
from peeler.dataset import PeelerDataset
from peeler.loss import SparseFocalBCELoss

__all__ = [
    'PurelyRelationalPeeler',
    'PeelerDataset',
    'SparseFocalBCELoss',
]
