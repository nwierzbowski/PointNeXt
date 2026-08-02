"""Validation functions for peeler training (GPU & Vectorized Accelerated)."""
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ThreadPoolExecutor

# Try using fastcluster if installed (drop-in 10x faster C++ replacement for scipy.cluster.hierarchy.linkage)
try:
    import fastcluster
    fast_linkage = fastcluster.linkage
except ImportError:
    fast_linkage = linkage

FRAGMENT_RANGES = [
    ((1, 2), '1'),
    ((2, 5), '2-4'),
    ((5, 12), '5-11'),
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

    # Fill diagonal with 1.0 (exact match to original code)
    A = b_affinities.copy()
    np.fill_diagonal(A, 1.0)

    # Condensed form & linkage
    condensed_dist = pdist(A, metric='cosine')
    Z = fast_linkage(condensed_dist, method='average')

    # Fast evaluation at unique merge heights
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

    # Initialize tracking structures
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
            mask = batch['mask'].to(device)
            Y = batch['Y'].to(device).float()
            asset_ids = batch['asset_ids'].to(device)

            soup_stats = batch.get('soup_stats', [])
            B, N = embeddings.shape[0], embeddings.shape[1]

            # Forward pass
            with torch.amp.autocast(device_type=device_type):
                affinity_logits = model(embeddings, transforms, mask)
                loss, _ = criterion(affinity_logits, asset_ids, mask)

            total_loss += loss.item() * B
            num_batches += 1
            total_samples += B

            # Compute probabilities on GPU
            affinity_probs = torch.sigmoid(affinity_logits)  # [B, N, N]
            soft_A = affinity_probs.cpu().numpy()             # [B, N, N] CPU NumPy array

            # ---------------------------------------------------------
            # 1. VECTORIZED F1 CALCULATION ON GPU
            # ---------------------------------------------------------
            mask_2d = (mask.unsqueeze(1) * mask.unsqueeze(2)) > 0.5  # [B, N, N]
            diag_mask = (1.0 - torch.eye(N, device=device)).unsqueeze(0) > 0.5  # [B, N, N]
            valid_pair_mask = mask_2d & diag_mask  # [B, N, N]

            # Reshape for broadcast against thresholds: [B, N, N, 100]
            preds = affinity_probs.unsqueeze(-1) > thresholds_tensor.view(1, 1, 1, -1)
            targets = (Y.unsqueeze(-1) > 0.5) & valid_pair_mask.unsqueeze(-1)

            tp_batch = (targets & preds).sum(dim=(1, 2)).cpu().numpy()  # [B, 100]
            fp_batch = ((~targets) & preds & valid_pair_mask.unsqueeze(-1)).sum(dim=(1, 2)).cpu().numpy()
            fn_batch = (targets & (~preds)).sum(dim=(1, 2)).cpu().numpy()

            asset_ids_np = asset_ids.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # Prepare items for CPU linkage worker
            cpu_tasks = []
            sample_keys = []

            for b in range(B):
                range_key = None
                if soup_stats and b < len(soup_stats):
                    actual_n = soup_stats[b].get('actual_n', 0)
                    range_key = _frag_to_range_key(actual_n)

                sample_keys.append(range_key)

                active_idx = np.where(mask_np[b] > 0.5)[0]
                if len(active_idx) <= 1:
                    cpu_tasks.append(None)
                else:
                    b_aff = soft_A[b][active_idx][:, active_idx]
                    b_gt = asset_ids_np[b][active_idx]
                    cpu_tasks.append((b_aff, b_gt))

            # ---------------------------------------------------------
            # 2. CPU ARI COMPUTATION (Threaded across batch items)
            # ---------------------------------------------------------
            def worker(task):
                if task is None:
                    return None
                return _eval_single_sample_ari(task[0], task[1], thresholds_np)

            with ThreadPoolExecutor() as executor:
                ari_results = list(executor.map(worker, cpu_tasks))

            # Accumulate metrics
            for b in range(B):
                range_key = sample_keys[b]
                track_bucket = range_key is not None and range_key in buckets

                # F1 accumulation
                tp_sums += tp_batch[b]
                fp_sums += fp_batch[b]
                fn_sums += fn_batch[b]

                if track_bucket:
                    bucket = buckets[range_key]
                    bucket['tp_sums'] += tp_batch[b]
                    bucket['fp_sums'] += fp_batch[b]
                    bucket['fn_sums'] += fn_batch[b]
                    bucket['loss_sum'] += loss.item()
                    bucket['sample_count'] += 1

                # ARI accumulation
                b_ari = ari_results[b]
                if b_ari is not None:
                    valid_sample_count_ari += 1
                    ari_sums += b_ari
                    if track_bucket:
                        buckets[range_key]['ari_sums'] += b_ari
                        buckets[range_key]['valid_sample_count_ari'] += 1

    # Finalize global metrics
    avg_loss = total_loss / max(total_samples, 1)

    # Compute best global ARI
    best_ari = -1.0
    best_ari_threshold = 0.5
    if valid_sample_count_ari > 0:
        avg_ari_per_t = ari_sums / valid_sample_count_ari
        best_t_idx = np.argmax(avg_ari_per_t)
        best_ari = float(avg_ari_per_t[best_t_idx])
        best_ari_threshold = float(thresholds_np[best_t_idx])

    # Compute best global F1
    precisions = np.where((tp_sums + fp_sums) > 0, tp_sums / (tp_sums + fp_sums), 0.0)
    recalls = np.where((tp_sums + fn_sums) > 0, tp_sums / (tp_sums + fn_sums), 0.0)
    f1_scores = np.where((precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls), 0.0)

    best_f1_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_f1_idx])
    best_f1_threshold = float(thresholds_np[best_f1_idx])

    # Compute bucket metrics
    bucket_metrics = {}
    for range_key, bucket in buckets.items():
        if bucket['sample_count'] == 0:
            continue

        b_loss = bucket['loss_sum'] / bucket['sample_count']

        # Bucket ARI
        b_best_ari = -1.0
        if bucket['valid_sample_count_ari'] > 0:
            b_avg_ari = bucket['ari_sums'] / bucket['valid_sample_count_ari']
            b_best_ari = float(np.max(b_avg_ari))

        # Bucket F1
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