"""Peeler model setup: build dataset, model, criterion, optimizer, scheduler.

All hyperparameters are read from the YAML config file.
Embeddings, transforms, and positions come from TBOManager data.
"""
import torch
from torch.utils.data import DataLoader

from openpoints.loss import build_criterion_from_cfg
from openpoints.models.build import build_model_from_cfg
from openpoints.utils import EasyConfig


def get_device(device):
    """Auto-detect device and wrap in torch.device."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def _build_peeler_dataset(cfg, all_embeddings, all_transforms):
    """Build PeelerDataset from TBO data.

    Args:
        cfg: EasyConfig with dataset settings
        all_embeddings: list of numpy arrays (N_i, 256), one per asset
        all_transforms: list of numpy arrays (N_i, 16), one per asset

    Returns:
        PeelerDataset instance
    """
    from openpoints.dataset.peeler_dataset import PeelerDataset
    dataset_cfg = cfg.dataset
    return PeelerDataset(
        all_embeddings=all_embeddings,
        all_transforms=all_transforms,
        max_assets_per_soup=dataset_cfg.get('max_assets_per_soup', 10),
        min_assets_per_soup=dataset_cfg.get('min_assets_per_soup', 2),
        max_asset_fragments=dataset_cfg.get('max_asset_fragments', None),
        seed=dataset_cfg.get('seed', 42),
    )


def _load_checkpoint(model, checkpoint_path, train=False, optimizer=None, scheduler=None, device=None):
    """Load checkpoint weights."""
    import os
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


def setup(
    all_embeddings,
    all_transforms,
    config_path,
    checkpoint_path=None,
    mode='train',
    yaml_content=None,
    log_callback=None,
    test_embeddings=None,
    test_transforms=None,
):
    """Unified setup for peeler training.

    Args:
        all_embeddings: list of numpy arrays (N_i, 256), one per asset
        all_transforms: list of numpy arrays (N_i, 16), one per asset
        config_path: path to YAML config file
        checkpoint_path: path to checkpoint file (for resume)
        mode: 'train'
        yaml_content: YAML config string from checkpoint
        log_callback: callable(str) for progress logging
        test_embeddings: list of numpy arrays (N_i, 256), one per test asset
        test_transforms: list of numpy arrays (N_i, 16), one per test asset

    Returns:
        (model, train_loader, val_loader, optimizer, scheduler, scaler, start_epoch, device, num_epochs)
    """
    # Load config
    if yaml_content is not None:
        import yaml
        cfg = EasyConfig()
        cfg.update(yaml.safe_load(yaml_content))
    else:
        cfg = EasyConfig()
        cfg.load(config_path, recursive=False)

    device = get_device(cfg.training.device)
    if log_callback:
        log_callback(f'Device: {device}')

    num_epochs = cfg.epochs
    batch_size = cfg.batch_size

    # Build train dataset
    train_dataset = _build_peeler_dataset(cfg, all_embeddings, all_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3,
        collate_fn=train_dataset.collate_fn,
    )
    if log_callback:
        log_callback(f'Train dataset: {len(train_loader.dataset)} samples')

    # Build validation dataset (optional)
    val_loader = None
    if test_embeddings is not None and test_transforms is not None:
        val_dataset = _build_peeler_dataset(cfg, test_embeddings, test_transforms)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=3,
            collate_fn=val_dataset.collate_fn,
        )
        if log_callback:
            log_callback(f'Val dataset: {len(val_loader.dataset)} samples')

    # Build model
    model = build_model_from_cfg(cfg.model)
    model = model.to(device)
    if log_callback:
        total_params = sum(p.numel() for p in model.parameters())
        log_callback(f'Model parameters: {total_params / 1e6:.2f}M')

    # Build criterion
    criterion = build_criterion_from_cfg(cfg.criterion)

    # Build optimizer
    optimizer_cfg = cfg.optimizer
    if optimizer_cfg.NAME == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=optimizer_cfg.get('weight_decay', 0.01),
        )
    elif optimizer_cfg.NAME == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=optimizer_cfg.get('weight_decay', 0.0),
        )
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_cfg.NAME}')

    # Build scheduler
    oc = cfg.onecycle if hasattr(cfg, 'onecycle') else {}
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=oc.get('pct_start', 0.3),
        anneal_strategy=oc.get('anneal_strategy', 'cos'),
        div_factor=oc.get('div_factor', 25),
        final_div_factor=oc.get('final_div_factor', 1e4),
    )

    # AMP scaler
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    model.train()

    # Load checkpoint state if resuming
    start_epoch = 0
    if checkpoint_path:
        if log_callback:
            log_callback(f'Loading checkpoint: {checkpoint_path}')
        start_epoch, _, _ = _load_checkpoint(model, checkpoint_path, train=True, optimizer=optimizer, scheduler=scheduler, device=device)
        if start_epoch > 0 and log_callback:
            log_callback(f'Resuming training from epoch {start_epoch}')

    return model, train_loader, val_loader, optimizer, scheduler, scaler, start_epoch, device, num_epochs
