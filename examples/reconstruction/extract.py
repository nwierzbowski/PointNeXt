"""Embedding extraction pipeline: setup and extract."""
import time

import torch


def extract(
    model,
    extract_loader,
    total,
    stop_callback=None,
    log_callback=None,
    progress_callback=None,
    save_callback=None,
):
    """Run embedding extraction inference loop.

    Args:
        model: PointNextMAE model instance (eval mode)
        extract_loader: DataLoader yielding dict with 'pos', 'x'
        total: total number of samples
        stop_callback: callable() -> bool, returns True if extraction should stop
        log_callback: callable(str) for progress logging
        progress_callback: callable(current, total) for batch progress
        save_callback: callable(uuids, embeddings_np) for batch DB persistence

    Returns:
        saved_count: int, number of embeddings saved
    """
    device = next(model.parameters()).device
    saved_count = 0

    # Timing accumulators
    gpu_transfer_time = 0.0
    inference_time = 0.0
    cpu_transfer_time = 0.0
    numpy_convert_time = 0.0
    db_save_time = 0.0
    batch_count = 0
    batch_intervals = 0.0
    prev_time = None

    with torch.no_grad():
        for batch in extract_loader:
            if stop_callback and stop_callback():
                if log_callback:
                    log_callback('Embedding extraction cancelled.')
                break

            if prev_time is not None:
                batch_intervals += time.time() - prev_time
            prev_time = time.time()

            t0 = time.time()
            p = batch['pos'].to(device, non_blocking=True)
            x = batch['x'].to(device, non_blocking=True)
            gpu_transfer_time += time.time() - t0

            t0 = time.time()
            embeddings = model.get_latent({'pos': p, 'x': x})
            inference_time += time.time() - t0

            t0 = time.time()
            cpu_tensor = embeddings.to('cpu', non_blocking=True)
            cpu_transfer_time += time.time() - t0

            t0 = time.time()
            embeddings_np = cpu_tensor.numpy()
            numpy_convert_time += time.time() - t0

            t0 = time.time()
            save_callback(batch['uuids'], embeddings_np)
            db_save_time += time.time() - t0

            saved_count += len(batch['uuids'])
            batch_count += 1

            if progress_callback:
                progress_callback(saved_count, total)
            if log_callback:
                log_callback(f'Processed {saved_count}/{total} assets')

    # Print cumulative timing summary
    if batch_count > 0:
        total_active = gpu_transfer_time + inference_time + cpu_transfer_time + numpy_convert_time + db_save_time
        total_wall = total_active + batch_intervals
        batch_size = extract_loader.batch_size
        print(f'\n{"="*60}')
        print(f'Embedding Extraction Timing Summary')
        print(f'{"="*60}')
        print(f'Batches processed: {batch_count}')
        print(f'Total samples: {saved_count}')
        print(f'Batch size: {batch_size}')
        print(f'{"-"*60}')
        phases = [
            ('GPU Transfer', gpu_transfer_time),
            ('Inference', inference_time),
            ('CPU Transfer (.cpu())', cpu_transfer_time),
            ('Numpy Convert (.numpy())', numpy_convert_time),
            ('DB Save', db_save_time),
        ]
        for name, t in phases:
            pct = (t / total_active * 100) if total_active > 0 else 0
            avg_ms = (t / batch_count) * 1000
            print(f'  {name:28s}: {t:8.2f}s ({pct:5.1f}%)  avg {avg_ms:7.1f}ms/batch')
        if batch_intervals > 0:
            pct = batch_intervals / total_wall * 100
            print(f'  {"DataLoading":28s}: {batch_intervals:8.2f}s ({pct:5.1f}%)  (DataLoader overhead)')
        print(f'{"-"*60}')
        print(f'  {"Total Active":28s}: {total_active:8.2f}s')
        print(f'  {"Total Wall":28s}: {total_wall:8.2f}s')
        print(f'  {"Samples/sec":28s}: {saved_count / total_wall:8.1f}')
        print(f'{"="*60}\n')

    return saved_count
