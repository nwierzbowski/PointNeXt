"""Post-training ARI analysis for peeler model.

Performs full sweep over topk values to evaluate model performance
across scene sizes using a single validation dataset per topk call.
The validate() function already tracks per-bucket metrics internally.
"""
from .validate import FRAGMENT_RANGES, validate


def run_analysis(model, val_loader, topk_values, device, criterion,
                 max_batches=None, progress_callback=None, edge_modes=['raw']):
    """Full ARI sweep analysis over edge_mode × topk × scene size buckets.

    Uses a single dataset call per topk — validate() already tracks per-bucket metrics.

    Args:
        model: Peeler model (eval mode)
        val_loader: DataLoader for validation set (with collate_fn)
        topk_values: list of topk values to sweep [1-32]
        device: device string
        criterion: loss function
        max_batches: optional limit per call
        progress_callback: optional (bucket, topk, best_ari) for real-time graph updates
        edge_modes: list of edge weight modes - default ['raw']

    Returns:
        dict with structure:
            {
                'buckets': {
                    'raw': {
                        '1': {'topk': {1: {...}, 2: {...}, ...}, 'scene_count': int},
                        ...
                    },
                    'gmean': {...},
                },
                'bucket_order': ['1', '2-11', ...],
                'evaluation': {
                    'ari_topk': {...},
                    'edge_mode': 'gmean',
                },
            }
    """

    model.eval()
    bucket_order = [item[1] for item in FRAGMENT_RANGES]

    results = {
        'buckets': {},
        'bucket_order': bucket_order,
        'evaluation': {},
    }

    for mode in edge_modes:
        # Initialize bucket structures for this mode
        results['buckets'][mode] = {}
        for bucket_key in bucket_order:
            results['buckets'][mode][bucket_key] = {'topk': {}, 'scene_count': 0}

        for topk in topk_values:
            val_results = validate(
                model, val_loader, criterion, device,
                max_batches=max_batches, ari_topk=topk, edge_mode=mode
            )

            # Extract per-bucket results from validate() return dict
            for bucket_key in bucket_order:
                if bucket_key not in val_results['bucket_metrics']:
                    continue

                bm = val_results['bucket_metrics'][bucket_key]
                results['buckets'][mode][bucket_key]['scene_count'] = bm['count']
                results['buckets'][mode][bucket_key]['topk'][topk] = {
                    'best_ari': bm['ari'],
                    'best_threshold': bm['threshold'],
                    'ari_curve': val_results['bucket_ari_curves'].get(bucket_key, []),
                    'avg_loss': bm['loss'],
                }

                if progress_callback:
                    progress_callback(bucket_key, topk, bm['ari'])

    # Compute best evaluation settings across all modes from buckets
    best_evaluation_topk = {}
    best_mode = None
    best_global_ari = -1.0

    for bucket_key in bucket_order:
        for mode in edge_modes:
            bucket_topk = results['buckets'][mode][bucket_key].get('topk', {})
            if not bucket_topk:
                continue
            best_topk = max(bucket_topk.keys(), key=lambda t: bucket_topk[t]['best_ari'])
            mode_ari = bucket_topk[best_topk]['best_ari']
            if mode_ari > best_global_ari or (mode_ari == best_global_ari and best_mode is None):
                best_global_ari = mode_ari
                best_mode = mode
            if bucket_key not in best_evaluation_topk:
                best_evaluation_topk[bucket_key] = {'ari': mode_ari, 'topk': best_topk, 'threshold': bucket_topk[best_topk]['best_threshold'], 'mode': mode}
            else:
                if mode_ari > best_evaluation_topk[bucket_key]['ari']:
                    best_evaluation_topk[bucket_key] = {'ari': mode_ari, 'topk': best_topk, 'threshold': bucket_topk[best_topk]['best_threshold'], 'mode': mode}

    # Build flat evaluation section with best settings
    evaluation_topk = {bucket_key: entry['topk'] for bucket_key, entry in best_evaluation_topk.items()}
    results['evaluation'] = {'ari_topk': evaluation_topk, 'edge_mode': best_mode}

    return results
