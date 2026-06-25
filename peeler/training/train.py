"""Peeler training pipeline: epoch loop and single epoch."""
import os
import time
import yaml
from pathlib import Path

import torch

from .utils import save_checkpoint
from .validate import validate

_TRAIN_YAML_PATH = None
_TRAIN_YAML_CONTENT = None


def peeler_train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    device,
    checkpoint_dir,
    start_epoch,
    log_callback=None,
    epoch_callback=None,
    step_callback=None,
    histogram_callback=None,
    stop_callback=None,
):
    """Full peeler training pipeline.

    Reads peeler.yaml for all config values, handles checkpoint saving internally.

    Args:
        model: Peeler model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: AMP scaler (None if not using AMP)
        criterion: Loss function
        device: Device string (e.g., 'cuda')
        checkpoint_dir: Directory for checkpoint files
        start_epoch: Epoch to resume from (0 = fresh start)
        log_callback: Called with (message: str)
        epoch_callback: Called with (epoch, total, loss, val_loss, val_ari, lr, val_f1)
        step_callback: Called with (step, epoch, loss)
        histogram_callback: Called with ((asset_counts: list[int], avg_frags: list[float])) after each epoch
        stop_callback: Called with no args, returns True to stop training

    Returns:
        (best_metric, best_metric_value, best_epoch)
    """
    global _TRAIN_YAML_PATH, _TRAIN_YAML_CONTENT

    # 1. Load peeler.yaml
    global _TRAIN_YAML_PATH
    if _TRAIN_YAML_PATH is None:
        _TRAIN_YAML_PATH = Path(__file__).parent.parent / 'peeler.yaml'

    with open(_TRAIN_YAML_PATH) as f:
        cfg = yaml.safe_load(f)

    # 2. Read yaml content for checkpoint embedding
    global _TRAIN_YAML_CONTENT
    if _TRAIN_YAML_CONTENT is None:
        with open(_TRAIN_YAML_PATH) as f:
            _TRAIN_YAML_CONTENT = f.read()

    # 3. Extract ALL config values (explicit, no defaults)
    num_epochs = cfg['epochs']
    device = device or cfg['training']['device']
    report_interval = cfg['training']['report_interval']
    ema_alpha = cfg['training']['ema_alpha']
    best_metric = 'ari'
    embedding_noise_sigma = cfg['embedding_noise_sigma']
    validation_threshold = cfg.get('validation', {}).get('threshold', 0.5)

    # 4. Internal train() — called with all config loaded
    return _train(
        model, train_loader, val_loader, optimizer, scheduler, device,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        criterion=criterion,
        scaler=scaler,
        start_epoch=start_epoch,
        report_interval=report_interval,
        ema_alpha=ema_alpha,
        best_metric=best_metric,
        embedding_noise_sigma=embedding_noise_sigma,
        validation_threshold=validation_threshold,
        save_checkpoint_callback=_make_checkpoint_callback(),
        log_callback=log_callback,
        epoch_callback=epoch_callback,
        step_callback=step_callback,
        histogram_callback=histogram_callback,
        stop_callback=stop_callback,
    )


def _make_checkpoint_callback():
    """Create checkpoint callback that embeds yaml_content."""
    def save_ckpt(model, opt, epoch, loss, path, scheduler=None,
                  best_metric=None, best_metric_value=None):
        save_checkpoint(model, opt, epoch, loss, path, scheduler,
                        yaml_content=_TRAIN_YAML_CONTENT,
                        best_metric=best_metric,
                        best_metric_value=best_metric_value)
    return save_ckpt


def _train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_dir,
    criterion,
    save_checkpoint_callback=None,
    log_callback=None,
    start_epoch=0,
    stop_callback=None,
    epoch_callback=None,
    scaler=None,
    step_callback=None,
    histogram_callback=None,
    report_interval=10,
    ema_alpha=0.1,
    best_metric='ari',
    embedding_noise_sigma=0.0,
    validation_threshold=0.5,
):
    """Run the training epoch loop."""
    best_metric_value = float('inf') if best_metric not in ('ari',) else -1.0
    best_epoch = 0
    global_step = 0
    ema_loss = None

    for epoch in range(start_epoch + 1, num_epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        epoch_start_time = time.time()

        model.train()
        total_loss = 0.0
        num_batches = 0
        stopped = False
        use_amp = scaler is not None
        soup_asset_counts = []
        soup_asset_fragments = []

        for batch_idx, batch in enumerate(train_loader):
            dtype = torch.bfloat16 if use_amp else torch.float32
            embeddings = batch['embeddings'].to(device, dtype=dtype, non_blocking=True)
            if embedding_noise_sigma > 0:
                embeddings = embeddings + torch.randn_like(embeddings) * embedding_noise_sigma
            transforms = batch['transforms'].to(device, dtype=dtype, non_blocking=True)
            mask = batch['mask'].to(device, dtype=torch.float32, non_blocking=True)
            asset_ids = batch['asset_ids'].to(device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                model_name = type(model).__name__
                if model_name == 'PurelyRelationalPeeler':
                    Y = batch['Y'].to(device, dtype=torch.float32, non_blocking=True)
                    affinity_logits = model(embeddings, transforms, mask)
                    loss, loss_dict = criterion(affinity_logits, Y, mask, epoch, num_epochs)
                else:
                    refined_emb = model(embeddings, transforms, mask)
                    loss, loss_dict = criterion(refined_emb, asset_ids, mask, epoch, num_epochs)

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

            batch_loss = loss.item()

            scheduler.step_update(global_step) if hasattr(scheduler, 'step_update') else scheduler.step()
            total_loss += batch_loss
            num_batches += 1
            global_step += 1

            # Collect soup stats for histogram
            if 'soup_stats' in batch:
                for stats in batch['soup_stats']:
                    soup_asset_counts.append(stats['num_assets'])
                    soup_asset_fragments.append(stats['asset_fragments'])

            if stop_callback and stop_callback():
                if log_callback:
                    log_callback('Training stopped by user')
                stopped = True
                break

            if step_callback and global_step % report_interval == 0:
                if ema_loss is None:
                    ema_loss = batch_loss
                else:
                    ema_loss = ema_alpha * batch_loss + (1 - ema_alpha) * ema_loss
                step_callback(global_step, epoch, ema_loss)

                if log_callback:
                    elapsed = time.time() - epoch_start_time
                    iters_per_sec = num_batches / elapsed if elapsed > 0 else 0
                    lr = scheduler.optimizer.param_groups[0]['lr'] if hasattr(scheduler, 'optimizer') else optimizer.param_groups[0]['lr']
                    log_callback(
                        f'Step {global_step} (epoch {epoch}) - '
                        f'Loss: {ema_loss:.4f} | LR: {lr:.6f} | '
                        f'{iters_per_sec:.1f} it/s'
                    )

        if stopped:
            break

        avg_loss = total_loss / max(num_batches, 1)

        # Validation every epoch
        val_loss = avg_loss
        val_ari = 0.0
        val_f1 = 0.0
        if val_loader is not None:
            val_loss, val_ari, val_f1 = validate(
                model, val_loader, criterion, device, scaler, epoch, num_epochs, threshold=validation_threshold
            )

        if epoch_callback:
            lr = optimizer.param_groups[0]['lr']
            epoch_callback(epoch, num_epochs, avg_loss, val_loss, val_ari, lr, val_f1)

        # Report asset count histogram
        if soup_asset_counts and histogram_callback:
            histogram_callback((tuple(soup_asset_counts), tuple(tuple(f) for f in soup_asset_fragments)))

        # Best model by selected metric
        metric_values = {
            'val_loss': val_loss,
            'train_loss': avg_loss,
            'ari': val_ari,
        }
        current_metric = metric_values[best_metric]
        is_better = current_metric < best_metric_value if best_metric not in ('ari',) else current_metric > best_metric_value

        if is_better:
            best_metric_value = current_metric
            best_epoch = epoch
            ckpt_path = os.path.join(checkpoint_dir, f'best_{best_metric}.pth')
            if save_checkpoint_callback:
                save_checkpoint_callback(model, optimizer, epoch, current_metric, ckpt_path, scheduler,
                                         best_metric=best_metric, best_metric_value=current_metric)
            else:
                save_checkpoint(model, optimizer, epoch, current_metric, ckpt_path, scheduler,
                                best_metric=best_metric, best_metric_value=current_metric)

        if epoch % 10 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            if save_checkpoint_callback:
                save_checkpoint_callback(model, optimizer, epoch, val_loss, ckpt_path, scheduler)
            else:
                save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path, scheduler)

    return best_metric, best_metric_value, best_epoch
