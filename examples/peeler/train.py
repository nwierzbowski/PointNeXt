"""Peeler training pipeline: setup, epoch loop, and single epoch."""
import os
import time

import torch

from openpoints.loss import build_criterion_from_cfg
from openpoints.models.build import build_model_from_cfg
from openpoints.dataset.build import build_dataset_from_cfg


def save_checkpoint(model, optimizer, epoch, loss, path, scheduler=None, yaml_content=None,
                    best_metric=None, best_metric_value=None):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'yaml_content': yaml_content,
        'best_metric': best_metric,
        'best_metric_value': best_metric_value,
    }
    torch.save(state, path)


def setup_model(cfg):
    """Build model from config."""
    model_cfg = cfg.model
    model = build_model_from_cfg(model_cfg)
    return model


def setup_criterion(cfg):
    """Build criterion from config."""
    return build_criterion_from_cfg(cfg.criterion)


def setup_optimizer(cfg, model):
    """Build optimizer from config."""
    optimizer_cfg = cfg.optimizer
    if optimizer_cfg.NAME == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=optimizer_cfg.get('weight_decay', 0.01),
        )
    elif optimizer_cfg.NAME == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=optimizer_cfg.get('weight_decay', 0.0),
        )
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_cfg.NAME}')


def setup_scheduler(cfg, optimizer, train_loader=None):
    """Build LR scheduler from config."""
    if cfg.sched == 'onecycle':
        oc = cfg.onecycle if hasattr(cfg, 'onecycle') else {}
        total_steps = cfg.epochs * len(train_loader) if train_loader else cfg.epochs * 100
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=oc.get('pct_start', 0.3),
            anneal_strategy=oc.get('anneal_strategy', 'cos'),
            div_factor=oc.get('div_factor', 25),
            final_div_factor=oc.get('final_div_factor', 1e4),
        )
    elif cfg.sched == 'warmup_cosine':
        from openpoints.scheduler import CosineAnnealingWarmupRestarts
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=cfg.epochs * 100,
            cycle_mult=1.,
            max_lr=cfg.lr,
            min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_t,
        )
    elif cfg.sched == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )
    else:
        return None


def train(
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
    soup_data_callback=None,
    report_interval=10,
    ema_alpha=0.1,
    best_metric='val_loss',
    embedding_noise_sigma=0.0,
):
    """Run the training epoch loop."""
    best_metric_value = float('inf') if best_metric != 'f1' else -1.0
    best_epoch = 0
    global_step = 0
    ema_loss = None
    ema_bce = None

    for epoch in range(start_epoch + 1, num_epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        epoch_start_time = time.time()

        model.train()
        total_loss = 0.0
        total_bce = 0.0
        num_batches = 0
        stopped = False
        use_amp = scaler is not None

        for batch_idx, batch in enumerate(train_loader):
            dtype = torch.bfloat16 if use_amp else torch.float32
            embeddings = batch['embeddings'].to(device, dtype=dtype, non_blocking=True)
            if embedding_noise_sigma > 0:
                embeddings = embeddings + torch.randn_like(embeddings) * embedding_noise_sigma
            transforms = batch['transforms'].to(device, dtype=dtype, non_blocking=True)
            mask = batch['mask'].to(device, dtype=torch.float32, non_blocking=True)
            Y = batch['Y'].to(device, dtype=dtype, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                anchor_probs, affinity_logits = model(embeddings, transforms, mask)
                loss, loss_dict = criterion(anchor_probs, affinity_logits, Y, mask)

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
            total_bce += loss_dict.get('loss_bce', batch_loss)
            num_batches += 1
            global_step += 1

            if stop_callback and stop_callback():
                if log_callback:
                    log_callback('Training stopped by user')
                stopped = True
                break

            if step_callback and global_step % report_interval == 0:
                if ema_loss is None:
                    ema_loss = batch_loss
                    ema_bce = loss_dict.get('loss_bce', batch_loss)
                else:
                    ema_loss = ema_alpha * batch_loss + (1 - ema_alpha) * ema_loss
                    ema_bce = ema_alpha * loss_dict.get('loss_bce', batch_loss) + (1 - ema_alpha) * ema_bce
                step_callback(global_step, epoch, ema_loss, ema_bce)

                # Emit soup data for viewport visualization
                if soup_data_callback:
                    seed_idx = torch.argmax(anchor_probs, dim=-1)
                    soup_data_callback(batch, anchor_probs, affinity_logits, model)
                if log_callback:
                    elapsed = time.time() - epoch_start_time
                    iters_per_sec = num_batches / elapsed if elapsed > 0 else 0
                    lr = scheduler.optimizer.param_groups[0]['lr'] if hasattr(scheduler, 'optimizer') else optimizer.param_groups[0]['lr']
                    log_callback(
                        f'Step {global_step} (epoch {epoch}) - '
                        f'BCE: {ema_bce:.4f} = Total: {ema_loss:.4f} | LR: {lr:.6f} | '
                        f'{iters_per_sec:.1f} it/s'
                    )

            # Free intermediate tensors to prevent GPU memory accumulation
            # del embeddings, transforms, mask, Y, loss, anchor_probs, affinity_logits
        if stopped:
            break

        # Release GPU memory between epochs
        torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        avg_bce = total_bce / max(num_batches, 1)

        # Validation every epoch
        val_loss = avg_loss
        val_bce = avg_bce
        val_acc = 0.0
        val_f1 = 0.0
        if val_loader is not None:
            val_loss, val_bce, val_acc, val_f1 = _validate(model, val_loader, criterion, device, scaler)

        if epoch_callback:
            lr = optimizer.param_groups[0]['lr']
            epoch_callback(epoch, num_epochs, avg_loss, val_loss, val_acc, val_f1, lr, avg_bce, val_bce)

        # Best model by selected metric
        metric_values = {
            'val_loss': val_loss,
            'train_loss': avg_loss,
            'f1': val_f1,
        }
        current_metric = metric_values[best_metric]
        is_better = current_metric < best_metric_value if best_metric != 'f1' else current_metric > best_metric_value

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


def _validate(model, val_loader, criterion, device, scaler):
    """Run validation loop and return loss, accuracy, and F1 score."""
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    correct = 0
    total = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for (batch_idx, batch) in enumerate(val_loader):
            embeddings = batch['embeddings'].to(device)
            transforms = batch['transforms'].to(device)
            mask = batch['mask'].to(device)
            Y = batch['Y'].to(device).float()

            anchor_probs, affinity_logits = model(embeddings, transforms, mask)
            loss, loss_dict = criterion(anchor_probs, affinity_logits, Y, mask)
            total_loss += loss.item()
            total_bce += loss_dict.get('loss_bce', loss.item())

            # Select best anchor via argmax (inference-time behavior)
            seed_idx = torch.argmax(anchor_probs, dim=-1)  # (B,)
            rows = torch.arange(Y.shape[0], device=Y.device)
            membership_logits = affinity_logits[rows, seed_idx]  # (B, N) - logits for selected anchor
            pred = torch.sigmoid(membership_logits) > 0.5
            Y_selected = Y[rows, seed_idx]  # (B, N)
            target = Y_selected > 0.5
            mask_expanded = mask.float()

            # Accuracy
            correct += ((pred == target).float() * mask_expanded).sum().item()
            total += mask_expanded.sum().item()

            # F1 components (per-fragment, binary)
            pred_mask = pred.float() * mask_expanded
            target_mask = target.float() * mask_expanded
            tp += ((pred_mask == 1) & (target_mask == 1)).sum().item()
            fp += ((pred_mask == 1) & (target_mask == 0)).sum().item()
            fn += ((pred_mask == 0) & (target_mask == 1)).sum().item()

    num_batches = max(len(val_loader), 1)
    avg_loss = total_loss / num_batches
    avg_bce = total_bce / num_batches
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, avg_bce, accuracy, f1
