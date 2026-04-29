"""Training utilities for PointNeXt MAE reconstruction.

This module provides high-level training functions for the PointNextMAE model,
including data preparation, model building, and the training loop.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

# Add PointNeXt root to path for openpoints imports
_POINTNEXT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _POINTNEXT_ROOT not in sys.path:
    sys.path.insert(0, _POINTNEXT_ROOT)


def train_mae_from_data(
    positions,
    features,
    uuids,
    config_path,
    checkpoint_dir,
    num_epochs=300,
    batch_size=32,
    lr=0.001,
    in_channels=6,
    num_points=1024,
    device=None,
    log_callback=None,
):
    """Train a PointNextMAE model from raw data.

    High-level entry point that handles dataset building, model loading,
    optimizer setup, and training loop.

    Args:
        positions: list of numpy arrays (N, 3)
        features: list of numpy arrays (N, C)
        uuids: list of string identifiers
        config_path: path to YAML config file
        checkpoint_dir: directory to save checkpoints
        num_epochs: number of training epochs
        batch_size: batch size for DataLoader
        lr: learning rate
        in_channels: number of input channels (3 + features)
        num_points: number of points per sample
        device: torch device ('cuda' or 'cpu')
        log_callback: callable(str) for progress logging

    Returns:
        best_loss: float, the best training loss achieved
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build dataset and dataloader
    dataset = _TBOMemoryDataset(positions, features, uuids, num_points, in_channels)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    if log_callback:
        log_callback(f'Train dataset: {len(dataset)} samples')

    # Build model
    model = _build_model(config_path, in_channels)
    model = model.to(device)

    if log_callback:
        total_params = sum(p.numel() for p in model.parameters())
        log_callback(f'Model parameters: {total_params / 1e6:.2f}M')

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5,
    )

    # Training loop
    if log_callback:
        log_callback(f'Starting training: {num_epochs} epochs')

    best_loss = train_mae(
        model,
        train_loader,
        optimizer,
        scheduler,
        device,
        num_epochs,
        checkpoint_dir,
        log_callback,
    )

    return best_loss


def train_mae(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_dir,
    log_callback=None,
):
    """Train a PointNextMAE model.

    Lower-level function that handles just the training loop.
    Useful when caller already has their own loader/optimizer.

    Args:
        model: PointNextMAE instance
        train_loader: DataLoader yielding dict with 'pos', 'x'
        optimizer: torch optimizer
        scheduler: LR scheduler
        device: torch device
        num_epochs: number of epochs
        checkpoint_dir: directory to save checkpoints
        log_callback: callable(str) for progress logging

    Returns:
        best_loss: float
    """
    model.train()
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            data = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            loss, pred, latent = model(data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if log_callback:
            log_callback(f'Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, 'best.pth')
            _save_checkpoint(
                model, optimizer, epoch, avg_loss, ckpt_path,
            )

        if epoch % 50 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            _save_checkpoint(
                model, optimizer, epoch, avg_loss, ckpt_path,
            )

    final_ckpt = os.path.join(checkpoint_dir, 'final.pth')
    _save_checkpoint(model, optimizer, num_epochs, best_loss, final_ckpt)

    return best_loss


def _build_model(config_path, in_channels):
    """Build a PointNextMAE model from config, with hardcoded fallback."""
    try:
        from openpoints.models import build_model_from_cfg
        from openpoints.utils import EasyConfig

        cfg = EasyConfig()
        cfg.load(config_path, recursive=True)

        # Override in_channels
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'encoder_args'):
            cfg.model.encoder_args.in_channels = in_channels

        model = build_model_from_cfg(cfg.model)
        return model
    except Exception:
        # Fallback: build model directly with hardcoded config
        from openpoints.models.reconstruction.pointnext_mae import PointNextMAE
        from openpoints.utils.config import EasyConfig

        encoder_args = {
            'NAME': 'PointNextEncoder',
            'blocks': [1, 1, 1, 1, 1, 1],
            'strides': [1, 2, 2, 2, 2, 1],
            'width': 32,
            'in_channels': in_channels,
            'sa_layers': 2,
            'sa_use_res': True,
            'radius': 0.15,
            'radius_scaling': 1.5,
            'nsample': 32,
            'expansion': 4,
            'aggr_args': {
                'feature_type': 'dp_fj',
                'reduction': 'max',
            },
            'group_args': {
                'NAME': 'ballquery',
                'normalize_dp': True,
            },
            'conv_args': {'order': 'conv-norm-act'},
            'act_args': {'act': 'relu'},
            'norm_args': {'norm': 'bn'},
        }

        encoder_cfg = EasyConfig()
        encoder_cfg.update(encoder_args)

        return PointNextMAE(
            encoder_args=encoder_cfg,
            latent_dim=256,
            decoder_points=1024,
            decoder_hidden_dim=512,
            jitter_sigma=0.01,
            jitter_prob=0.9,
        )


class _TBOMemoryDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for in-memory point cloud data."""

    def __init__(
        self,
        positions,
        features,
        uuids,
        num_points=1024,
        in_channels=6,
    ):
        self.positions = positions
        self.features = features
        self.uuids = uuids
        self.num_points = num_points
        self.in_channels = in_channels

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        pos = self.positions[idx].astype('float32')
        feat = self.features[idx].astype('float32')

        # Check for NaN/Inf in raw data — replace with zeros
        pos_mask = np.isnan(pos) | np.isinf(pos)
        if pos_mask.any():
            pos = np.where(pos_mask, 0.0, pos)
        feat_mask = np.isnan(feat) | np.isinf(feat)
        if feat_mask.any():
            feat = np.where(feat_mask, 0.0, feat)

        n = len(pos)
        if n != self.num_points:
            raise ValueError(
                f"Sample {self.uuids[idx]} has {n} points but expected {self.num_points}. "
                "All data must already be resized to num_points before passing to the dataset."
            )

        pos = torch.from_numpy(pos)
        feat = torch.from_numpy(feat)

        return {
            'pos': pos,
            'x': self._build_x(pos, feat),
        }

    def _build_x(self, pos, feat):
        """Build input features tensor."""
        if self.in_channels > 3:
            needed = self.in_channels - 3
            if feat.shape[1] < needed:
                pad = torch.zeros(
                    feat.shape[0], needed - feat.shape[1],
                    dtype=torch.float32, device=feat.device,
                )
                feat = torch.cat([feat, pad], dim=1)
            return torch.cat([pos, feat[:, :needed]], dim=1)
        return pos


def _save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        path,
    )
