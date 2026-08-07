"""Post-training ARI analysis for peeler model.

Performs full sweep over topk values to evaluate model performance
across scene sizes using a single validation dataset per topk call.
The validate() function already tracks per-bucket metrics internally.
"""
import time

from .validate import FRAGMENT_RANGES, validate


def run_analysis(model, val_loader, topk_values, device, criterion,
                 max_batches=None, log_callback=None, progress_callback=None):
    """Full ARI sweep analysis over topk × scene size buckets.

    Uses a single dataset call per topk — validate() already tracks per-bucket metrics.

    Args:
        model: Peeler model (eval mode)
        val_loader: DataLoader for validation set (with collate_fn)
        topk_values: list of topk values to sweep [1-32]
        device: device string
        criterion: loss function
        max_batches: optional limit per call
        log_callback: optional (msg: str) for progress logging
        progress_callback: optional (bucket, topk, best_ari) for real-time updates

    Returns:
        dict with structure:
            {
                'buckets': {
                    '1': {'topk': {1: {...}, 2: {...}, ...}, 'scene_count': int},
                    '2-11': {...},
                    ...
                },
                'bucket_order': ['1', '2-11', ...]
            }
    """
    model.eval()
    bucket_order = [item[1] for item in FRAGMENT_RANGES]
    total_calls = len(topk_values)

    results = {'buckets': {}, 'bucket_order': bucket_order}

    # Initialize bucket structures
    for bucket_key in bucket_order:
        results['buckets'][bucket_key] = {'topk': {}, 'scene_count': 0}

    if log_callback:
        log_callback(f'Running {total_calls} validation calls (one per topk)...')
        log_callback('')

    call_count = 0
    for topk in topk_values:
        call_count += 1
        call_start = time.time()

        val_results = validate(
            model, val_loader, criterion, device,
            max_batches=max_batches, ari_topk=topk
        )

        call_elapsed = time.time() - call_start

        # Extract per-bucket results from validate() return dict
        for bucket_key in bucket_order:
            if bucket_key not in val_results['bucket_metrics']:
                continue

            bm = val_results['bucket_metrics'][bucket_key]
            results['buckets'][bucket_key]['scene_count'] = bm['count']
            results['buckets'][bucket_key]['topk'][topk] = {
                'best_ari': bm['ari'],
                'best_threshold': bm['threshold'],
                'ari_curve': val_results['bucket_ari_curves'].get(bucket_key, []),
                'avg_loss': bm['loss'],
            }

            if progress_callback:
                progress_callback(bucket_key, topk, bm['ari'])

            if log_callback:
                log_callback(
                    f'  [{call_count}/{total_calls}] {bucket_key:>7}: '
                    f'TopK={topk:>2}, ARI={bm["ari"]:.4f} (threshold={bm["threshold"]:.2f}) '
                    f'[{call_elapsed:.1f}s]'
                )

        if log_callback:
            log_callback(f'  [{call_count}/{total_calls}] TopK={topk}: Complete in {call_elapsed:.1f}s')

    # Compute summary: best per bucket across all topk values
    best_per_bucket = {}
    bucket_aris = []
    for bucket_key in bucket_order:
        bucket_topk = results['buckets'][bucket_key].get('topk', {})
        if not bucket_topk:
            continue
        best_topk = max(bucket_topk.keys(), key=lambda t: bucket_topk[t]['best_ari'])
        best_entry = bucket_topk[best_topk]
        best_per_bucket[bucket_key] = {
            'ari': best_entry['best_ari'],
            'threshold': best_entry['best_threshold'],
            'topk': best_topk,
        }
        bucket_aris.append(best_entry['best_ari'])

    global_avg_ari = sum(bucket_aris) / len(bucket_aris) if bucket_aris else 0.0

    results['summary'] = {
        'global_avg_ari': round(global_avg_ari, 4),
        'best_per_bucket': best_per_bucket,
    }

    # Build evaluation section for apply-to-config
    evaluation_topk = {}
    for bucket_key, entry in best_per_bucket.items():
        evaluation_topk[bucket_key] = entry['topk']
    results['evaluation'] = {'ari_topk': evaluation_topk}

    if log_callback:
        log_callback(f'Best topk per bucket: {evaluation_topk}')

    return results
