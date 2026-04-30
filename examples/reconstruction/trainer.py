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
    device = _get_device(device)

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
    if resume_from:
        if os.path.exists(resume_from):
            start_epoch = _load_training_checkpoint(resume_from, model, optimizer, scheduler, device)
            if log_callback:
                log_callback(f'Loaded checkpoint: {resume_from} (epoch {start_epoch})')
        else:
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
        in_channels,
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
    in_channels=6,
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
            data = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'uuids'}

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                loss, pred, latent = model(data)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
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
                model, optimizer, epoch, avg_loss, ckpt_path, scheduler, in_channels,
            )

        if epoch % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            _save_checkpoint(
                model, optimizer, epoch, avg_loss, ckpt_path, scheduler, in_channels,
            )

    return best_loss


def _get_device(device):
    """Auto-detect device and wrap in torch.device.

    Args:
        device: 'cuda', 'cpu', None, or torch.device.

    Returns:
        torch.device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def _read_checkpoint_in_channels(checkpoint_path):
    """Read in_channels from checkpoint metadata without loading the model.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        int or None if not found.
    """
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        for key in ['model', 'net', 'network', 'state_dict', 'base_model']:
            if key in state_dict:
                state_dict = state_dict[key]
        return state_dict.get('in_channels', None)
    except Exception:
        return None


def _clean_data(pos, feat):
    """Replace NaN/Inf values with zeros.

    Args:
        pos: (N, 3) numpy array.
        feat: (N, C) numpy array.

    Returns:
        Tuple of (cleaned_pos, cleaned_feat) as numpy arrays.
    """
    pos_mask = np.isnan(pos) | np.isinf(pos)
    if pos_mask.any():
        pos = np.where(pos_mask, 0.0, pos)
    feat_mask = np.isnan(feat) | np.isinf(feat)
    if feat_mask.any():
        feat = np.where(feat_mask, 0.0, feat)
    return pos, feat


def _build_x_numpy(pos, feat, in_channels):
    """Build input features tensor from numpy arrays.

    Pads or truncates feature channels to match in_channels, then
    concatenates with positions.

    Args:
        pos: (N, 3) numpy array of positions.
        feat: (N, C) numpy array of features.
        in_channels: Total input channels (3 + features).

    Returns:
        torch.Tensor of shape (N, in_channels).
    """
    if in_channels > 3:
        needed = in_channels - 3
        if feat.shape[1] < needed:
            pad = np.zeros((feat.shape[0], needed - feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=1)
        feat = feat[:, :needed]
        return torch.from_numpy(np.concatenate([pos, feat], axis=1)).float()
    return torch.from_numpy(pos).float()


def _build_x_tensor(pos, feat, in_channels):
    """Build input features tensor from torch tensors.

    Pads or truncates feature channels to match in_channels, then
    concatenates with positions.

    Args:
        pos: (N, 3) torch tensor.
        feat: (N, C) torch tensor.
        in_channels: Total input channels (3 + features).

    Returns:
        torch.Tensor of shape (N, in_channels).
    """
    if in_channels > 3:
        needed = in_channels - 3
        if feat.shape[1] < needed:
            pad = torch.zeros(
                feat.shape[0], needed - feat.shape[1],
                dtype=torch.float32, device=feat.device,
            )
            feat = torch.cat([feat, pad], dim=1)
        feat = feat[:, :needed]
        return torch.cat([pos, feat], dim=1)
    return pos


def _build_model(config_path, in_channels=None):
    """Build a PointNextMAE model from YAML config.

    Args:
        config_path: Path to YAML config file.
        in_channels: Number of input channels. If None, uses config value.
                    If provided, overrides config value.

    Returns:
        PointNextMAE model instance.
    """
    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig

    cfg = EasyConfig()
    cfg.load(config_path, recursive=False)

    # Override in_channels if provided (from checkpoint metadata)
    if in_channels is not None and hasattr(cfg, 'model') and hasattr(cfg.model, 'encoder_args'):
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

        # Clean NaN/Inf in raw data
        pos, feat = _clean_data(pos, feat)

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
            'x': _build_x_tensor(pos, feat, self.in_channels),
            'uuids': self.uuids[idx],
        }


def _save_checkpoint(model, optimizer, epoch, loss, path, scheduler=None, in_channels=None):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'in_channels': in_channels,
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)


def _load_checkpoint(model, checkpoint_path):
    """Load trained weights into model with strict matching.

    Handles different checkpoint formats (nested under 'model', 'net', etc.)
    and removes 'module.' prefix from DDP-saved weights.

    Raises:
        RuntimeError: If checkpoint architecture doesn't match model (strict mode).

    Args:
        model: PointNextMAE model instance.
        checkpoint_path: Path to checkpoint file.

    Returns:
        Tuple of (None, in_channels) where in_channels is the value stored in checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    ckpt_state_dict = state_dict
    for key in ['model', 'net', 'network', 'state_dict', 'base_model', 'model_state_dict']:
        if key in state_dict:
            ckpt_state_dict = state_dict[key]

    # Read in_channels from checkpoint metadata
    in_channels = ckpt_state_dict.get('in_channels', None)

    base_ckpt = {k.replace('module.', ''): v for k, v in ckpt_state_dict.items()}
    model.load_state_dict(base_ckpt, strict=True)
    return None, in_channels


def _load_training_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint for training resume.

    Uses _load_checkpoint for model weights (with format/DDP handling),
    then loads optimizer, scheduler, and epoch metadata.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: PointNextMAE model instance.
        optimizer: torch optimizer.
        scheduler: LR scheduler.
        device: torch device.

    Returns:
        start_epoch: int, epoch to resume from (0 if fresh training).
    """
    # Load model weights with full format/DDP handling
    _, _ = _load_checkpoint(model, checkpoint_path)

    # Load optimizer/scheduler/epoch from checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    for key in ['model', 'net', 'network', 'state_dict', 'base_model', 'model_state_dict']:
        if key in state_dict:
            state_dict = state_dict[key]

    start_epoch = state_dict.get('epoch', 0)
    if 'optimizer_state_dict' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    if 'scheduler_state_dict' in state_dict:
        try:
            scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        except Exception:
            pass

    return start_epoch


def _batch_save(uuids, embeddings_np, save_callback):
    """Pass batch through to callback — callback handles iteration.

    Args:
        uuids: List of mesh hash strings.
        embeddings_np: (batch_size, dim) numpy array.
        save_callback: callable(uuids, embeddings_np).
    """
    save_callback(uuids, embeddings_np)


def _preprocess_sample(pos, feat, in_channels):
    """Preprocess a single sample: NaN/Inf handling + feature tensor building.

    Args:
        pos: (N, 3) numpy array of positions.
        feat: (N, C) numpy array of features.
        in_channels: Total input channels (3 + features).

    Returns:
        torch.Tensor of shape (N, in_channels) with positions + features.
    """
    pos, feat = _clean_data(pos, feat)
    return _build_x_numpy(pos, feat, in_channels)


def extract_latent_from_data(
    positions,
    features,
    uuids,
    config_path,
    checkpoint_path,
    num_points=1024,
    batch_size=128,
    device=None,
    stop_callback=None,
    log_callback=None,
    progress_callback=None,
    save_callback=None,
):
    """Extract embeddings from raw point cloud data.

    High-level entry point that handles model loading, batch processing
    via DataLoader, and embedding extraction.

    Args:
        positions: list of numpy arrays (N, 3)
        features: list of numpy arrays (N, C)
        uuids: list of string identifiers (mesh hashes)
        config_path: path to YAML config file
        checkpoint_path: path to trained checkpoint file
        num_points: number of points per sample
        batch_size: batch size for extraction
        device: torch device ('cuda' or 'cpu'), or None for auto-detect
        stop_callback: callable() -> bool, returns True if extraction should stop
        log_callback: callable(str) for progress logging
        progress_callback: callable(current, total) for batch progress
        save_callback: callable(uuids, embeddings_np) for batch DB persistence

    Returns:
        saved_count: int, number of embeddings saved
    """
    device = _get_device(device)

    if log_callback:
        log_callback(f'Loading model from: {checkpoint_path}')
        log_callback(f'Device: {device}')

    # Read in_channels from checkpoint metadata first
    ckpt_in_channels = _read_checkpoint_in_channels(checkpoint_path)
    if ckpt_in_channels is not None:
        if log_callback:
            log_callback(f'Checkpoint in_channels: {ckpt_in_channels}')
    else:
        if log_callback:
            log_callback('Warning: Checkpoint has no in_channels metadata')

    # Build model with checkpoint's in_channels
    if log_callback:
        log_callback('Building PointNeXt model...')
    model = _build_model(config_path, ckpt_in_channels)

    # Load checkpoint
    _, _ = _load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    if log_callback:
        log_callback(f'Loaded checkpoint: {checkpoint_path}')

    # Build dataset and DataLoader for extraction
    dataset = _TBOMemoryDataset(positions, features, uuids, num_points, ckpt_in_channels or 6)
    extract_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    if log_callback:
        log_callback(f'Extraction dataset: {len(dataset)} samples')

    # Extract embeddings in batches
    total = len(positions)
    saved_count = 0

    with torch.no_grad():
        for batch in extract_loader:
            if stop_callback and stop_callback():
                if log_callback:
                    log_callback('Embedding extraction cancelled.')
                return saved_count

            p = batch['pos'].to(device, non_blocking=True)
            x = batch['x'].to(device, non_blocking=True)

            embeddings = model.get_latent({'pos': p, 'x': x})
            embeddings_np = embeddings.cpu().numpy()

            _batch_save(batch['uuids'], embeddings_np, save_callback)
            saved_count += len(batch['uuids'])

            if progress_callback:
                progress_callback(saved_count, total)
            if log_callback:
                log_callback(f'Processed {saved_count}/{total} assets')

    return saved_count
