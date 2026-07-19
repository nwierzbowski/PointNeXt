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
    validation_threshold = cfg.get('validation', {}).get('threshold', 0.5)

    # 4. Internal train() — called with all config loaded
    grad_accum_steps = cfg.get('grad_accum_steps', 1)

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
        validation_threshold=validation_threshold,
        grad_accum_steps=grad_accum_steps,
        save_checkpoint_callback=_make_checkpoint_callback(),
        log_callback=log_callback,
        epoch_callback=epoch_callback,
        step_callback=step_callback,
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
    report_interval=10,
    ema_alpha=0.1,
    best_metric='ari',
    validation_threshold=0.5,
    grad_accum_steps=1,
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

        step_loss_accum = 0.0
        for batch_idx, batch in enumerate(train_loader):
            dtype = torch.bfloat16 if use_amp else torch.float32
            embeddings = batch['embeddings'].to(device, dtype=dtype, non_blocking=True)
            transforms = batch['transforms'].to(device, dtype=dtype, non_blocking=True)
            mask = batch['mask'].to(device, dtype=torch.float32, non_blocking=True)
            asset_ids = batch['asset_ids'].to(device, dtype=torch.long, non_blocking=True)

            is_accum_start = batch_idx % grad_accum_steps == 0
            if is_accum_start:
                optimizer.zero_grad()
                step_loss_accum = 0.0

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                Y = batch['Y'].to(device, dtype=torch.float32, non_blocking=True)
                out_embeddings = model(embeddings, transforms, mask)
                loss, loss_dict = criterion(out_embeddings, Y, mask)

            step_loss_accum += loss.item()

            loss = loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accum_end = (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(train_loader) - 1
            if is_accum_end:
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                # print(f"[GRAD] grad_norm={grad_norm_before_clip:.4f}")
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step_update(global_step) if hasattr(scheduler, 'step_update') else scheduler.step()
                global_step += 1

                actual_accum = (batch_idx + 1) % grad_accum_steps
                if actual_accum == 0:
                    actual_accum = grad_accum_steps
                batch_loss = step_loss_accum / actual_accum

                total_loss += batch_loss
                num_batches += 1

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

            if stop_callback and stop_callback():
                if log_callback:
                    log_callback('Training stopped by user')
                stopped = True
                break

        if stopped:
            break

        avg_loss = total_loss / max(num_batches, 1)

        # Validation every epoch
        val_loss = avg_loss
        val_ari = 0.0
        val_f1 = 0.0
        if val_loader is not None:
            val_loss, val_ari, val_f1, ari_thres, f1_thres = validate(
                model, val_loader, criterion, device, scaler, epoch, num_epochs, threshold=validation_threshold
            )

        if epoch_callback:
            lr = optimizer.param_groups[0]['lr']
            epoch_callback(epoch, num_epochs, avg_loss, val_loss, val_ari, lr, val_f1, ari_thres, f1_thres)

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
