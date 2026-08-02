"""Peeler training pipeline: epoch loop and single epoch."""
import os
import time
import yaml
from pathlib import Path

import torch

from ..curriculum import CURRICULUM_BUCKETS
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
    Runs indefinitely until stopped via stop_callback.

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
        epoch_callback: Called with (log_message: str)
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

    # 3. Extract config values
    device = device or cfg['training']['device']
    report_interval = cfg['training']['report_interval']
    best_metric = 'ari'

    # 4. Internal train() — called with all config loaded
    grad_accum_steps = cfg.get('grad_accum_steps', 1)
    batch_size_per_phase = cfg.get('batch_size_per_phase', [])
    grad_accum_steps_per_phase = cfg.get('grad_accum_steps_per_phase', [])
    max_batches = cfg.get('validation', {}).get('max_batches')

    return _train(
        model, train_loader, val_loader, optimizer, scheduler, device,
        checkpoint_dir=checkpoint_dir,
        criterion=criterion,
        scaler=scaler,
        start_epoch=start_epoch,
        report_interval=report_interval,
        best_metric=best_metric,
        grad_accum_steps=grad_accum_steps,
        batch_size_per_phase=batch_size_per_phase,
        grad_accum_steps_per_phase=grad_accum_steps_per_phase,
        max_batches=max_batches,
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
    best_metric='ari',
    grad_accum_steps=1,
    batch_size_per_phase=None,
    grad_accum_steps_per_phase=None,
    max_batches=None,
):
    """Run the training epoch loop (infinite, competence-based curriculum)."""
    best_metric_value = float('inf') if best_metric not in ('ari',) else -1.0
    best_epoch = 0
    global_step = 0
    report_loss = 0.0
    graph_loss = 0.0
    original_base_wd = optimizer.param_groups[0].get('weight_decay', 0.1)
    epoch = start_epoch

    num_phases = len(CURRICULUM_BUCKETS)

    def _resolve_per_phase(per_phase_list, fallback):
        if not per_phase_list:
            return [fallback] * num_phases
        if len(per_phase_list) < num_phases:
            per_phase_list = list(per_phase_list) + [per_phase_list[-1]] * (num_phases - len(per_phase_list))
        return per_phase_list[:num_phases]

    while True:
        epoch += 1
        train_loader.dataset.set_epoch(epoch, scheduler.phase, scheduler.ramp_progress)
        curriculum_phase_label = scheduler.phase_label
        effective_grad_accum = _resolve_per_phase(grad_accum_steps_per_phase, grad_accum_steps)[scheduler.phase]
        
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

            is_accum_start = batch_idx % effective_grad_accum == 0
            if is_accum_start:
                optimizer.zero_grad()
                step_loss_accum = 0.0

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                out_embeddings = model(embeddings, transforms, mask)
                loss, loss_dict = criterion(out_embeddings, asset_ids, mask)

            step_loss_accum += loss.item()

            loss = loss / effective_grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accum_end = (batch_idx + 1) % effective_grad_accum == 0 or batch_idx == len(train_loader) - 1
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

                global_step += 1

                actual_accum = (batch_idx + 1) % effective_grad_accum
                if actual_accum == 0:
                    actual_accum = effective_grad_accum
                batch_loss = step_loss_accum / actual_accum

                total_loss += batch_loss
                report_loss += batch_loss
                graph_loss += batch_loss
                num_batches += 1

                graph_interval = 5
                if step_callback and global_step % graph_interval == 0:
                    step_callback(global_step, epoch, graph_loss / graph_interval)
                    graph_loss = 0

                if log_callback and global_step % report_interval == 0:
                    elapsed = time.time() - epoch_start_time
                    iters_per_sec = num_batches / elapsed if elapsed > 0 else 0
                    lr = optimizer.param_groups[0]['lr']
                    log_callback(
                        f'Step {global_step} (epoch {epoch}) - '
                        f'Loss: {report_loss / report_interval:.4f} | LR: {lr:.6f} | '
                        f'{iters_per_sec:.1f} it/s'
                    )
                    report_loss = 0
                
                

            if stop_callback and stop_callback():
                if log_callback:
                    log_callback('Training stopped by user')
                stopped = True
                break

        if stopped:
            break

        avg_loss = total_loss / max(num_batches, 1)

        # Validation every epoch
        val_loss = -1
        val_ari = 0.0
        val_f1 = 0.0
        val_ari_thres = 0.0
        val_f1_thres = 0.0
        val_bucket_metrics = {}
        if val_loader is not None:
            val_results = validate(
                model, val_loader, criterion, device
            )
            val_loss = val_results['avg_loss']
            val_ari = val_results['best_ari']
            val_f1 = val_results['best_f1']
            val_ari_thres = val_results['best_ari_threshold']
            val_f1_thres = val_results['best_f1_threshold']
            val_bucket_metrics = val_results.get('bucket_metrics', {})

        # Training set validation
        train_results = validate(
            model, train_loader, criterion, device, max_batches=max_batches
        )
        train_loss = train_results['avg_loss']
        train_ari = train_results['best_ari']
        train_f1 = train_results['best_f1']
        train_ari_thres = train_results['best_ari_threshold']
        train_f1_thres = train_results['best_f1_threshold']

        # Step LR scheduler (per-epoch) — reports ARI, advances phase, steps LR
        train_bucket_ari = train_results.get('bucket_metrics', {})
        scheduler.step(train_bucket_ari)

        if epoch_callback:
            lr = optimizer.param_groups[0]['lr']
            # Build per-bucket summary strings
            def format_bucket_metrics(bucket_metrics):
                if not bucket_metrics:
                    return None
                entries = []
                for range_key in ['1', '2-4', '5-11', '12-26', '27-51', '52-101', '102-300', '300+']:
                    if range_key in bucket_metrics:
                        bm = bucket_metrics[range_key]
                        entries.append((range_key, bm))
                if not entries:
                    return None
                return entries

            val_buckets = format_bucket_metrics(val_bucket_metrics)
            train_buckets = format_bucket_metrics(train_results.get('bucket_metrics', {}))

            # Overall metrics line
            phase_str = f' | Phase: {curriculum_phase_label}' if curriculum_phase_label else ''
            log_msg = f'Epoch {epoch}{phase_str} | LR: {lr:.2e} | Loss: {avg_loss:.4f}\n'
            log_msg += f'  Val: Loss={val_loss:.3f} ARI={val_ari:.3f} F1={val_f1:.3f} (ARI@{val_ari_thres:.2f} F1@{val_f1_thres:.2f})\n'
            log_msg += f'  Trn: Loss={train_loss:.3f} ARI={train_ari:.3f} F1={train_f1:.3f} (ARI@{train_ari_thres:.2f} F1@{train_f1_thres:.2f})'

            # Bucket metrics - side by side val/train, one per line
            all_buckets = []
            for range_key in ['1', '2-4', '5-11', '12-26', '27-51', '52-101', '102-300', '300+']:
                val_bm = val_buckets and any(k == range_key for k, _ in val_buckets)
                train_bm = train_buckets and any(k == range_key for k, _ in train_buckets)
                if val_bm or train_bm:
                    all_buckets.append((range_key,
                                        val_buckets and next((bm for k, bm in val_buckets if k == range_key), None),
                                        train_buckets and next((bm for k, bm in train_buckets if k == range_key), None)))

            if all_buckets:
                log_msg += '\n  Buckets:\n'
                for range_key, v_bm, t_bm in all_buckets:
                    parts = [f'    Frag {range_key:>7} (n={v_bm["count"] if v_bm else t_bm["count"]:>4}):']
                    if v_bm:
                        parts.append(f'V ARI={v_bm["ari"]:.3f} F1={v_bm["f1"]:.3f}')
                    if t_bm:
                        parts.append(f'T ARI={t_bm["ari"]:.3f} F1={t_bm["f1"]:.3f}')
                    log_msg += ' ' + '  '.join(parts) + '\n'

            epoch_callback(log_msg.rstrip())

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
