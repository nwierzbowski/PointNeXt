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


def build_peeler_dataset(cfg, all_embeddings, all_transforms):
    """Build PeelerDataset from TBO data.

    Args:
        cfg: EasyConfig with dataset settings
        all_embeddings: list of numpy arrays (N_i, 256), one per asset
        all_transforms: list of numpy arrays (N_i, 16), one per asset

    Returns:
        PeelerDataset instance
    """
    from peeler.dataset import PeelerDataset
    dataset_cfg = cfg.dataset
    return PeelerDataset(
        all_embeddings=all_embeddings,
        all_transforms=all_transforms,
        max_fragments=dataset_cfg.get('max_fragments', 700),
        seed=dataset_cfg.get('seed', 42),
        translation_scale=dataset_cfg.get('translation_scale', 0.0),
        cluster_translation_scale=dataset_cfg.get('cluster_translation_scale', [1.6, 0.7]),
        asset_scale_std=dataset_cfg.get('asset_scale_std', 1.0),
        scene_scale=dataset_cfg.get('scene_scale', [0.8, 1.2]),
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
        (model, train_loader, val_loader, optimizer, scheduler, scaler, criterion, start_epoch, device, num_epochs)
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
    grad_accum_steps = cfg.get('grad_accum_steps', 1)

    # Build train dataset
    train_dataset = build_peeler_dataset(cfg, all_embeddings, all_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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
        val_dataset = build_peeler_dataset(cfg, test_embeddings, test_transforms)
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

    # Build optimizer with parameter grouping
    # Exclude biases and LayerNorm parameters from weight decay
    optimizer_cfg = cfg.optimizer
    weight_decay = optimizer_cfg.get('weight_decay', 0.01)

    no_decay_weight_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_weight_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_weight_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_cfg.NAME == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    elif optimizer_cfg.NAME == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=cfg.lr)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_cfg.NAME}')

    # Build scheduler (OneCycleLR with warmup + cosine decay)
    scheduler_cfg = cfg.get('scheduler', {})
    effective_steps_per_epoch = (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=num_epochs,
        steps_per_epoch=effective_steps_per_epoch,
        pct_start=cfg.lr_pct_start,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=scheduler_cfg.get('div', 25.0),
        final_div_factor=scheduler_cfg.get('final_div_factor', 10000.0),
    )

    # Attach num_epochs to criterion (used by loss.py for curriculum ramp)
    criterion._num_epochs = num_epochs

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

    return model, train_loader, val_loader, optimizer, scheduler, scaler, criterion, start_epoch, device, num_epochs
