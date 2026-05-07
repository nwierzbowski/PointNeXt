"""Training pipeline: setup, epoch loop, and single epoch."""
import os
import time

import torch


def save_checkpoint(model, optimizer, epoch, loss, path, scheduler=None, in_channels=None, yaml_content=None):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'in_channels': in_channels,
        'yaml_content': yaml_content,
    }
    torch.save(state, path)


def train(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_dir,
    in_channels,
    save_checkpoint_callback=None,
    log_callback=None,
    start_epoch=0,
    stop_callback=None,
    epoch_callback=None,
    scaler=None,
    step_callback=None,
    report_interval=10,
    reconstruction_callback=None,
    initial_global_step=0,
):
    """Run the training epoch loop.

    Args:
        model: PointNextMAE instance
        train_loader: DataLoader yielding dict with 'pos', 'x'
        optimizer: torch optimizer
        scheduler: LR scheduler
        device: torch device
        num_epochs: number of epochs
        checkpoint_dir: directory to save checkpoints
        in_channels: number of input channels (for checkpoint metadata)
        save_checkpoint_callback: callable(model, opt, epoch, loss, path, scheduler, in_channels)
        log_callback: callable(str) for progress logging
        start_epoch: epoch to start from (0 = fresh training)
        stop_callback: callable() -> bool, returns True if training should stop
        epoch_callback: callable(epoch, total_epochs, loss) for epoch progress
        scaler: GradScaler for AMP (None = no AMP)
        step_callback: callable(step, epoch, loss) for step progress
        report_interval: report loss every N batches
        reconstruction_callback: callable(uuid, positions, predictions) for reconstruction visualization
        initial_global_step: global step to start from (for resume)

    Returns:
        best_loss: float
    """
    best_loss = float('inf')
    global_step = initial_global_step

    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_start_time = time.time()
        report_loss_sum = 0.0
        report_batch_count = 0

        avg_loss, global_step, stopped, last_batch, last_pred = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            scaler,
            report_interval,
            log_callback,
            step_callback,
            reconstruction_callback,
            stop_callback,
            global_step,
            epoch_start_time,
            report_loss_sum,
            report_batch_count,
        )

        if stopped:
            break

        if epoch_callback:
            epoch_callback(epoch, num_epochs, avg_loss)
        if reconstruction_callback and last_batch is not None:
            data = {k: v.to(device, non_blocking=True) for k, v in last_batch.items() if k != 'uuids'}
            uuid = last_batch['uuids'][0] if isinstance(last_batch['uuids'], (list, tuple)) else last_batch['uuids'].item()
            positions = data['pos'][0].cpu().numpy()
            features = data.get('feat', None)
            if features is not None:
                gt_features = features[0].detach().float().cpu().numpy()
            else:
                gt_features = None
            predictions = last_pred[0].detach().float().cpu().numpy()
            reconstruction_callback(uuid, positions, predictions, gt_features)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, 'best.pth')
            save_checkpoint_callback(model, optimizer, epoch, avg_loss, ckpt_path, scheduler, in_channels)

        if epoch % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            save_checkpoint_callback(model, optimizer, epoch, avg_loss, ckpt_path, scheduler, in_channels)

    return best_loss


def run_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    scaler,
    report_interval,
    log_callback,
    step_callback,
    reconstruction_callback,
    stop_callback,
    global_step,
    start_time,
    report_loss_sum,
    report_batch_count,
):
    """Run one training epoch.

    Args:
        model: PointNextMAE instance
        train_loader: DataLoader yielding dict with 'pos', 'x'
        optimizer: torch optimizer
        scheduler: LR scheduler
        device: torch device
        epoch: current epoch number
        scaler: GradScaler for AMP (None = no AMP)
        report_interval: report loss every N batches
        log_callback: callable(str) for progress logging
        step_callback: callable(step, epoch, loss) for step progress
        reconstruction_callback: callable(uuid, positions, predictions) for visualization
        stop_callback: callable() -> bool, returns True if training should stop
        global_step: current global step count
        start_time: epoch start time (time.time())
        report_loss_sum: accumulated loss for current report interval
        report_batch_count: batch count for current report interval

    Returns:
        Tuple of (avg_loss, global_step, stopped, last_batch, last_pred)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    stopped = False
    use_amp = scaler is not None
    last_avg_batch_loss = 0.0

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

        batch_loss = loss.item()
        total_loss += batch_loss
        report_loss_sum += batch_loss
        report_batch_count += 1
        num_batches += 1
        global_step += 1
        scheduler.step_update(global_step)

        if stop_callback and stop_callback():
            if log_callback:
                log_callback('Training stopped by user')
            stopped = True
            break

        last_batch = batch
        last_pred = pred

        if step_callback and global_step % report_interval == 0:
            last_avg_batch_loss = report_loss_sum / report_batch_count
            step_callback(global_step, epoch, last_avg_batch_loss)
            if log_callback:
                elapsed = time.time() - start_time
                iters_per_sec = num_batches / elapsed if elapsed > 0 else 0
                lr = scheduler.optimizer.param_groups[0]['lr']
                log_callback(f'Step {global_step} (epoch {epoch}) - Avg Loss: {last_avg_batch_loss:.4f} | LR: {lr:.6f} | {iters_per_sec:.1f} it/s')
            report_loss_sum = 0.0
            report_batch_count = 0

        if reconstruction_callback and global_step % report_interval == 0:
            uuid = batch['uuids'][0] if isinstance(batch['uuids'], (list, tuple)) else batch['uuids'].item()
            positions = data['pos'][0].cpu().numpy()
            features = data.get('feat', None)
            if features is not None:
                gt_features = features[0].detach().float().cpu().numpy()
            else:
                gt_features = None
            predictions = pred[0].detach().float().cpu().numpy()
            reconstruction_callback(uuid, positions, predictions, gt_features)

    if stopped:
        return total_loss, global_step, stopped, last_batch, last_pred

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step, stopped, last_batch, last_pred
