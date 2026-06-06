"""Model construction, checkpoint loading, and unified setup for training/extraction.

All hyperparameters are read from the YAML config file.
Only in_channels and channel_indices come from TBO data.
"""
import os
import yaml

import torch
from torch.utils.data import DataLoader

from openpoints.dataset import build_dataset_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.utils import EasyConfig


def get_device(device):
    """Auto-detect device and wrap in torch.device.

    Args:
        device: 'cuda', 'cpu', None, or torch.device.

    Returns:
        torch.device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def _load_checkpoint(model, checkpoint_path, train=False, optimizer=None, scheduler=None, device=None):
    """Load checkpoint weights.

    Handles different checkpoint formats (nested under 'model', 'net', etc.)
    and removes 'module.' prefix from DDP-saved weights.

    Args:
        model: PointNextMAE model instance.
        checkpoint_path: Path to checkpoint file.
        train: If True, also load optimizer and epoch for resume.
        optimizer: torch optimizer (only used if train=True).
        scheduler: LR scheduler (only used if train=True, passed for compatibility).
        device: torch device (only used if train=True).

    Returns:
        If train=False: None
        If train=True: (start_epoch, yaml_content) tuple
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    full_state = torch.load(checkpoint_path, map_location=device if train else 'cpu')

    # Handle different checkpoint formats
    ckpt_state_dict = full_state
    for key in ['model', 'net', 'network', 'state_dict', 'base_model', 'model_state_dict']:
        if key in full_state:
            ckpt_state_dict = full_state[key]

    base_ckpt = {k.replace('module.', ''): v for k, v in ckpt_state_dict.items()}
    model.load_state_dict(base_ckpt, strict=True)

    if train:
        if 'optimizer_state_dict' in full_state:
            optimizer.load_state_dict(full_state['optimizer_state_dict'])
        start_epoch = full_state.get('epoch', 0)
        yaml_content = full_state.get('yaml_content')
        checkpoint_loss = full_state.get('loss', None)
        return start_epoch, yaml_content, checkpoint_loss

    return None


def _build_model(config, in_channels, channel_indices=None):
    """Build a PointNextMAE model from YAML config.

    Args:
        config: Path string or EasyConfig object.
        in_channels: Number of input channels (computed from TBO data).
        channel_indices: dict of pre-computed integer indices for feat tensor access.

    Returns:
        PointNextMAE model instance (with random weights).
    """
    if in_channels is None:
        raise ValueError('in_channels is required')

    if isinstance(config, str):
        cfg = EasyConfig()
        cfg.load(config, recursive=False)
    else:
        cfg = config

    # Override in_channels (pad 3 to 4 for vec4 alignment)
    effective_in_channels = in_channels
    if effective_in_channels == 3:
        effective_in_channels = 4
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'encoder_args'):
        cfg.model.encoder_args.in_channels = effective_in_channels

    # Pass pre-computed channel indices to model
    if channel_indices is not None:
        cfg.model.channel_indices = channel_indices

    model = build_model_from_cfg(cfg.model)
    return model


def setup(
    positions,
    features,
    uuids,
    config_path,
    checkpoint_path=None,
    mode='train',
    in_channels=None,
    channel_indices=None,
    yaml_content=None,
    log_callback=None,
    model=None,
    optimizer=None,
    scheduler=None,
):
    """Unified setup for training or extraction.

    All hyperparameters are read from the YAML config file.
    Only in_channels and channel_indices come from TBO data.

    Accepts pre-built model/optimizer/scheduler from model_manager.
    If not provided, builds them using openpoints cfg building.
    Loads checkpoint state if checkpoint_path is provided.

    Args:
        positions: list of numpy arrays (N, 3)
        features: list of numpy arrays (N, C)
        uuids: list of string identifiers
        config_path: path to YAML config file
        checkpoint_path: path to checkpoint file (for extract mode or resume)
        mode: 'train' or 'extract'
        in_channels: number of input channels (required, from TBO data)
        channel_indices: dict of pre-computed integer indices for feat tensor access
        yaml_content: YAML config string from checkpoint (for model building)
        log_callback: callable(str) for progress logging
        model: pre-built model (from model_manager.load_for_training, etc.)
        optimizer: pre-built optimizer (from model_manager)
        scheduler: pre-built scheduler (from model_manager)

    Returns:
        Training: (model, train_loader, optimizer, scheduler, scaler, start_epoch, initial_global_step, device, num_epochs, checkpoint_loss)
        Extract: (model, extract_loader, total_count)
    """
    if in_channels is None:
        raise ValueError('in_channels is required')

    # Use checkpoint YAML if provided, otherwise load from config_path
    if yaml_content is not None:
        cfg = EasyConfig()
        cfg.update(yaml.safe_load(yaml_content))
    else:
        cfg = EasyConfig()
        cfg.load(config_path, recursive=False)

    device = get_device(cfg.training.device)

    if log_callback:
        log_callback(f'Device: {device}')

    # Get training params from YAML config
    num_epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr

    # Build dataset from config + runtime TBO data
    dataset_cfg = cfg.dataset
    dataset = build_dataset_from_cfg(
        dataset_cfg,
        default_args={
            'positions': positions,
            'features': features,
            'uuids': uuids,
            'in_channels': in_channels,
            'encoder_indices': channel_indices.get('encoder_indices', None) if channel_indices else None,
        }
    )

    # Build dataloader (different for train vs extract)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=mode == 'train',
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3,
        collate_fn=dataset.collate_fn,
    )
    if log_callback:
        log_callback(f'{mode.title()} dataset: {len(loader.dataset)} samples')

    # Use pre-built model or build from config
    if model is None:
        model = _build_model(cfg, in_channels=in_channels, channel_indices=channel_indices)
        model = model.to(device)
        if log_callback:
            total_params = sum(p.numel() for p in model.parameters())
            log_callback(f'Model parameters: {total_params / 1e6:.2f}M')

    # Mode-specific setup
    if mode == 'train':
        # Calculate total steps for scheduler
        batches_per_epoch = len(loader.dataset) // batch_size
        total_steps = num_epochs * batches_per_epoch

        # Use pre-built optimizer or create via openpoints cfg building
        if optimizer is None:
            optimizer_cfg = cfg.optimizer
            optimizer = build_optimizer_from_cfg(model, lr=lr, **optimizer_cfg)

        # Use pre-built scheduler or create via openpoints cfg building
        if scheduler is None:
            cfg.t_max = total_steps
            scheduler = build_scheduler_from_cfg(cfg, optimizer)

        # AMP scaler
        use_amp = device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        model.train()

        # Load checkpoint state if resuming (optimizer + epoch)
        start_epoch = 0
        checkpoint_loss = None
        if checkpoint_path:
            if log_callback:
                log_callback(f'Loading checkpoint: {checkpoint_path}')
            start_epoch, _, checkpoint_loss = _load_checkpoint(model, checkpoint_path, train=True, optimizer=optimizer, scheduler=scheduler, device=device)
            if start_epoch > 0:
                if log_callback:
                    log_callback(f'Resuming training from epoch {start_epoch}')
            else:
                if log_callback:
                    log_callback(f'Checkpoint has no epoch metadata — starting fresh')
        else:
            if log_callback:
                log_callback(f'Starting training: {num_epochs} epochs')

        initial_global_step = start_epoch * len(loader)
        return model, loader, optimizer, scheduler, scaler, start_epoch, initial_global_step, device, num_epochs, checkpoint_loss

    else:  # extract
        if checkpoint_path:
            if log_callback:
                log_callback(f'Loading model from: {checkpoint_path}')
            _load_checkpoint(model, checkpoint_path)

        model.eval()

        if log_callback:
            log_callback(f'Extraction dataset: {len(loader.dataset)} samples')

        return model, loader, len(positions)
