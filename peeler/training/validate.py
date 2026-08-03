"""Validation functions for peeler training (GPU & Vectorized Accelerated)."""
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from concurrent.futures import ThreadPoolExecutor

try:
    import fastcluster
    fast_linkage = fastcluster.linkage
except ImportError:
    fast_linkage = linkage

FRAGMENT_RANGES = [
    ((1, 2), '1'),
    ((2, 12), '2-11'),
    ((12, 27), '12-26'),
    ((27, 52), '27-51'),
    ((52, 102), '52-101'),
    ((102, 301), '102-300'),
    ((301, float('inf')), '300+'),
]


def _frag_to_range_key(frag: int) -> str:
    """Map fragment count to range key string."""
    n = len(FRAGMENT_RANGES)
    for i, ((low, high), range_key) in enumerate(FRAGMENT_RANGES):
        if i == n - 1:
            if low <= frag <= high:
                return range_key
        else:
            if low <= frag < high:
                return range_key
    return 'other'


def _eval_single_sample_ari(b_affinities, gt_labels, thresholds):
    """Compute ARI across thresholds."""
    N = len(gt_labels)
    if N <= 1:
        return None

    A = b_affinities.copy()
    np.fill_diagonal(A, 1.0)

    # Protect against NaN/Inf in affinity matrix
    A = np.nan_to_num(A, nan=0.5, posinf=1.0, neginf=0.0)

    # Protect against zero-norm rows (cosine distance undefined)
    norms = np.linalg.norm(A, axis=1)
    zero_mask = norms < 1e-10
    A[zero_mask] = 0.0
    np.fill_diagonal(A, 1.0)  # Re-fill diagonal after zeroing

    condensed_dist = pdist(A, metric='cosine')
    condensed_dist = np.nan_to_num(condensed_dist, nan=2.0)

    Z = fast_linkage(condensed_dist, method='average')

    merge_heights = Z[:, 2]
    k_indices = np.searchsorted(merge_heights, thresholds, side='right')
    unique_ks = np.unique(k_indices)
    ari_per_k = {}

    for k in unique_ks:
        if k == 0:
            pred_labels = np.arange(N)
        else:
            t_sample = merge_heights[k - 1] if k < len(merge_heights) else merge_heights[-1] + 1e-5
            pred_labels = fcluster(Z, t=t_sample, criterion='distance')

        ari_per_k[k] = adjusted_rand_score(gt_labels, pred_labels)

    return np.array([ari_per_k[k] for k in k_indices])


def validate(model, val_loader, criterion, device, max_batches=None):
    """Run validation loop, sweeping over thresholds to find the best ARI and F1."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_samples = 0
    valid_sample_count_ari = 0

    thresholds_np = np.arange(0.0, 1.0, 0.01)
    thresholds_tensor = torch.from_numpy(thresholds_np).to(device, dtype=torch.float32)  # [100]
    num_thresholds = len(thresholds_np)

    ari_sums = np.zeros(num_thresholds, dtype=np.float64)
    tp_sums = np.zeros(num_thresholds, dtype=np.int64)
    fp_sums = np.zeros(num_thresholds, dtype=np.int64)
    fn_sums = np.zeros(num_thresholds, dtype=np.int64)

    buckets = {}
    for (low, high), range_key in FRAGMENT_RANGES:
        buckets[range_key] = {
            'ari_sums': np.zeros(num_thresholds, dtype=np.float64),
            'tp_sums': np.zeros(num_thresholds, dtype=np.int64),
            'fp_sums': np.zeros(num_thresholds, dtype=np.int64),
            'fn_sums': np.zeros(num_thresholds, dtype=np.int64),
            'loss_sum': 0.0,
            'sample_count': 0,
            'valid_sample_count_ari': 0,
        }

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            embeddings = batch['embeddings'].to(device)
            transforms = batch['transforms'].to(device)
            asset_ids = batch['asset_ids'].to(device)
            scene_ids = batch.get('scene_ids', None)
            mask = batch.get('mask', None)

            if scene_ids is not None:
                scene_ids = scene_ids.to(device)
            if mask is not None:
                mask = mask.to(device)
            else:
                mask = torch.ones(embeddings.shape[0], device=device)

            soup_stats = batch.get('soup_stats', [])

            with torch.amp.autocast(device_type=device_type):
                scene_logits = model(embeddings, transforms, scene_ids, mask)
                loss, _ = criterion(scene_logits, asset_ids, scene_ids, mask)

            if isinstance(scene_logits, dict):
                scenes = scene_logits
            else:
                scenes = {0: scene_logits}

            # Accumulate loss weighted by scene count
            num_scenes_in_batch = len(scenes)
            total_loss += loss.item() * num_scenes_in_batch
            num_batches += 1

            asset_ids_np = asset_ids.cpu().numpy()
            scene_ids_np = scene_ids.cpu().numpy() if scene_ids is not None else np.zeros(len(asset_ids), dtype=int)
            mask_np = mask.cpu().numpy()

            scene_data = []
            for scene_idx, logits in scenes.items():
                scene_node_mask = (scene_ids_np == scene_idx)
                all_scene_idx = np.where(scene_node_mask)[0]
                active_idx = np.where(scene_node_mask & (mask_np > 0.5))[0]

                # Direct scene_idx lookup for soup_stats
                range_key = None
                if soup_stats and scene_idx < len(soup_stats):
                    actual_n = soup_stats[scene_idx].get('actual_n', 0)
                    range_key = _frag_to_range_key(actual_n)

                global_to_local = {g: l for l, g in enumerate(all_scene_idx)}
                local_active = np.array([global_to_local[g] for g in active_idx])

                soft_A_full = torch.sigmoid(logits).cpu().numpy()
                soft_A = soft_A_full[local_active][:, local_active]
                scene_asset_ids_np = asset_ids_np[active_idx]
                scene_logits = logits[local_active][:, local_active]

                scene_data.append({
                    'scene_idx': scene_idx,
                    'soft_A': soft_A,
                    'gt_labels': scene_asset_ids_np,
                    'active_count': len(active_idx),
                    'range_key': range_key,
                    'logits': scene_logits,
                })

            def worker(sd):
                if sd['active_count'] <= 1:
                    return None
                return _eval_single_sample_ari(sd['soft_A'], sd['gt_labels'], thresholds_np)

            with ThreadPoolExecutor() as executor:
                ari_results = list(executor.map(worker, scene_data))

            for i, sd in enumerate(scene_data):
                range_key = sd['range_key']
                track_bucket = range_key is not None and range_key in buckets
                N_s = sd['active_count']

                if N_s <= 1:
                    continue

                total_samples += 1

                # GPU Vectorized F1 Computation
                probs = torch.sigmoid(sd['logits'])
                diag_mask = (1.0 - torch.eye(N_s, device=device)) > 0.5

                preds = probs.unsqueeze(-1) > thresholds_tensor.view(1, 1, -1)
                scene_aids_t = torch.from_numpy(sd['gt_labels']).to(device)
                targets = (scene_aids_t.unsqueeze(1) == scene_aids_t.unsqueeze(0)).unsqueeze(-1) & diag_mask.unsqueeze(-1)

                tp_scene = (targets & preds).sum(dim=(0, 1)).cpu().numpy()
                fp_scene = ((~targets) & preds & diag_mask.unsqueeze(-1)).sum(dim=(0, 1)).cpu().numpy()
                fn_scene = (targets & (~preds)).sum(dim=(0, 1)).cpu().numpy()

                tp_sums += tp_scene
                fp_sums += fp_scene
                fn_sums += fn_scene

                if track_bucket:
                    bucket = buckets[range_key]
                    bucket['tp_sums'] += tp_scene
                    bucket['fp_sums'] += fp_scene
                    bucket['fn_sums'] += fn_scene
                    bucket['loss_sum'] += loss.item()
                    bucket['sample_count'] += 1

                b_ari = ari_results[i]
                if b_ari is not None:
                    valid_sample_count_ari += 1
                    ari_sums += b_ari
                    if track_bucket:
                        buckets[range_key]['ari_sums'] += b_ari
                        buckets[range_key]['valid_sample_count_ari'] += 1

    # Finalize global metrics
    avg_loss = total_loss / max(total_samples, 1)

    # Best global ARI
    best_ari = -1.0
    best_ari_threshold = 0.5
    if valid_sample_count_ari > 0:
        avg_ari_per_t = ari_sums / valid_sample_count_ari
        best_t_idx = np.argmax(avg_ari_per_t)
        best_ari = float(avg_ari_per_t[best_t_idx])
        best_ari_threshold = float(thresholds_np[best_t_idx])

    # Best global F1
    precisions = np.where((tp_sums + fp_sums) > 0, tp_sums / (tp_sums + fp_sums), 0.0)
    recalls = np.where((tp_sums + fn_sums) > 0, tp_sums / (tp_sums + fn_sums), 0.0)
    f1_scores = np.where((precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls), 0.0)

    best_f1_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_f1_idx])
    best_f1_threshold = float(thresholds_np[best_f1_idx])

    # Bucket metrics
    bucket_metrics = {}
    for range_key, bucket in buckets.items():
        if bucket['sample_count'] == 0:
            continue

        b_loss = bucket['loss_sum'] / bucket['sample_count']

        b_best_ari = -1.0
        if bucket['valid_sample_count_ari'] > 0:
            b_avg_ari = bucket['ari_sums'] / bucket['valid_sample_count_ari']
            b_best_ari = float(np.max(b_avg_ari))

        b_tp, b_fp, b_fn = bucket['tp_sums'], bucket['fp_sums'], bucket['fn_sums']
        b_prec = np.where((b_tp + b_fp) > 0, b_tp / (b_tp + b_fp), 0.0)
        b_rec = np.where((b_tp + b_fn) > 0, b_tp / (b_tp + b_fn), 0.0)
        b_f1_scores = np.where((b_prec + b_rec) > 0, 2 * b_prec * b_rec / (b_prec + b_rec), 0.0)
        b_best_f1 = float(np.max(b_f1_scores))

        bucket_metrics[range_key] = {
            'loss': round(b_loss, 4),
            'ari': round(b_best_ari, 4),
            'f1': round(b_best_f1, 4),
            'count': bucket['sample_count'],
        }

    return {
        'avg_loss': round(avg_loss, 4),
        'best_ari': round(best_ari, 4),
        'best_f1': round(best_f1, 4),
        'best_ari_threshold': round(best_ari_threshold, 2),
        'best_f1_threshold': round(best_f1_threshold, 2),
        'bucket_metrics': bucket_metrics,
    }