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
    warmup_epochs=0,
    resume_from=None,
    stop_callback=None,
    device=None,
    log_callback=None,
    epoch_callback=None,
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
        warmup_epochs: number of warmup epochs
        resume_from: path to checkpoint to resume from
        stop_callback: callable() -> bool, returns True if training should stop
        device: torch device ('cuda' or 'cpu')
        log_callback: callable(str) for progress logging
        epoch_callback: callable(epoch, total_epochs, loss) for epoch progress

    Returns:
        best_loss: float, the best training loss achieved
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

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

    # Optimizer and scheduler (must be created before loading checkpoint)
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

    # AMP scaler for bfloat16 mixed precision
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Resume from checkpoint if provided (after optimizer/scheduler are created)
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        try:
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception:
                    pass
            if log_callback:
                log_callback(f'Loaded checkpoint: {resume_from} (epoch {start_epoch})')
        except Exception as e:
            if log_callback:
                log_callback(f'Warning: Could not load checkpoint: {e}')
            start_epoch = 0
    elif resume_from:
        if log_callback:
            log_callback(f'Warning: Checkpoint not found: {resume_from}')

    # Apply warmup if fresh training
    if start_epoch == 0 and warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5
                ),
            ],
            milestones=[warmup_epochs],
        )

    # Training loop
    if log_callback:
        if start_epoch > 0:
            log_callback(f'Resuming training from epoch {start_epoch}')
        else:
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
        start_epoch,
        stop_callback,
        epoch_callback,
        scaler,
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
    start_epoch=0,
    stop_callback=None,
    epoch_callback=None,
    scaler=None,
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
        start_epoch: epoch to start from (0 = fresh training)
        stop_callback: callable() -> bool, returns True if training should stop
        epoch_callback: callable(epoch, total_epochs, loss) for epoch progress
        scaler: GradScaler for AMP (None = no AMP)

    Returns:
        best_loss: float
    """
    model.train()
    best_loss = float('inf')
    stopped = False
    use_amp = scaler is not None

    for epoch in range(start_epoch + 1, num_epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            data = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                loss, pred, latent = model(data)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if stop_callback and stop_callback():
                if log_callback:
                    log_callback('Training stopped by user')
                stopped = True
                break

        if stopped:
            break

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if log_callback:
            log_callback(f'Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}')
        if epoch_callback:
            epoch_callback(epoch, num_epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, 'best.pth')
            _save_checkpoint(
                model, optimizer, epoch, avg_loss, ckpt_path, scheduler,
            )

        if epoch % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            _save_checkpoint(
                model, optimizer, epoch, avg_loss, ckpt_path, scheduler,
            )

    return best_loss


def _build_model(config_path, in_channels):
    """Build a PointNextMAE model from YAML config."""
    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig

    cfg = EasyConfig()
    cfg.load(config_path, recursive=False)

    # Override in_channels from UI
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'encoder_args'):
        cfg.model.encoder_args.in_channels = in_channels

    model = build_model_from_cfg(cfg.model)
    return model


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


def _save_checkpoint(model, optimizer, epoch, loss, path, scheduler=None):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)
