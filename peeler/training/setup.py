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


def build_peeler_dataset(cfg, all_embeddings, all_transforms, asset_to_file=None, mode='train'):
    """Build PeelerDataset for train or val.

    Args:
        cfg: EasyConfig with dataset settings
        all_embeddings: list of numpy arrays (N_i, 256), one per asset
        all_transforms: list of numpy arrays (N_i, 16), one per asset
        asset_to_file: list mapping asset_idx -> scene_idx (scene = TBO file)
        mode: 'train' or 'val'

    Returns:
        PeelerDataset instance
    """
    from peeler.dataset import PeelerDataset
    dataset_cfg = cfg.dataset
    dataset = PeelerDataset(
        mode=mode,
        all_embeddings=all_embeddings,
        all_transforms=all_transforms,
        asset_to_file=asset_to_file,
        max_fragments=dataset_cfg.get('max_fragments', 700),
        seed=dataset_cfg.get('seed', 42),
    )
    return dataset


def _load_checkpoint(model, checkpoint_path, train=False, optimizer=None, device=None):
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
    yaml_content=None,
    log_callback=None,
    asset_to_file=None,
    test_embeddings=None,
    test_transforms=None,
    test_asset_to_file=None,
):
    """Unified setup for peeler training.

    Args:
        all_embeddings: list of numpy arrays (N_i, 256), one per asset
        all_transforms: list of numpy arrays (N_i, 16), one per asset
        config_path: path to YAML config file
        checkpoint_path: path to checkpoint file (for resume)
        yaml_content: YAML config string from checkpoint
        log_callback: callable(str) for progress logging
        asset_to_file: list mapping train asset_idx -> scene_idx (scene = TBO file)
        test_embeddings: list of numpy arrays (N_i, 256), one per test asset
        test_transforms: list of numpy arrays (N_i, 16), one per test asset
        test_asset_to_file: list mapping test asset_idx -> file_idx for validation soups

    Returns:
        (model, train_loader, val_loader, optimizer, scheduler, scaler, criterion, start_epoch, device)
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

    batch_size = cfg.batch_size
    num_epochs = cfg.get('num_epochs', 200)
    lr_pct_start = cfg.get('lr_pct_start', 0.1)
    eta_min = cfg.get('eta_min', 1e-5)
    grad_accum_steps = cfg.get('grad_accum_steps', 1)
    warmup_epochs = max(1, int(num_epochs * lr_pct_start))

    # Build train dataset
    train_dataset = build_peeler_dataset(cfg, all_embeddings, all_transforms, asset_to_file, mode='train')
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
        if hasattr(train_loader.dataset, 'get_bucket_stats'):
            log_callback(train_loader.dataset.get_bucket_stats())

    # Build validation dataset (optional)
    val_loader = None
    if test_embeddings is not None and test_transforms is not None and test_asset_to_file is not None:
        val_dataset = build_peeler_dataset(cfg, test_embeddings, test_transforms, test_asset_to_file, mode='val')
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.validation.get('batch_size', 1),
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

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Do not apply weight decay to biases:
        if ".bias" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

    if optimizer_cfg.NAME == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)
    elif optimizer_cfg.NAME == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=cfg.lr)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_cfg.NAME}')

    # Build LR scheduler: linear warmup + cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / warmup_epochs, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=eta_min
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    if log_callback:
        log_callback(f'LR schedule: {warmup_epochs} warmup + {num_epochs - warmup_epochs} cosine, eta_min={eta_min}')

    # AMP scaler
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    model.train()

    # Load checkpoint state if resuming
    start_epoch = 0
    if checkpoint_path:
        if log_callback:
            log_callback(f'Loading checkpoint: {checkpoint_path}')
        start_epoch, _, _ = _load_checkpoint(model, checkpoint_path, train=True, optimizer=optimizer, device=device)
        if start_epoch > 0 and log_callback:
            log_callback(f'Resuming training from epoch {start_epoch}')

    return model, train_loader, val_loader, optimizer, scheduler, scaler, criterion, start_epoch, device
