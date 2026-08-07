"""Validation functions for peeler training (sparse Top-K competition).

Sparse Kruskal Union-Find Single-Linkage Clustering on sparse (N x K) graph edges.
Achieves O(N * K log(N*K)) runtime and zero dense N^2 memory allocations.
Accelerated via fused Numba C-kernel ARI evaluation.
"""
import numpy as np
import torch
from numba import njit


FRAGMENT_RANGES = [
    ((1, 2), '1'),
    ((2, 12), '2-11'),
    ((12, 27), '12-26'),
    ((27, 52), '27-51'),
    ((52, 102), '52-101'),
    ((102, 301), '102-300'),
    ((301, float('inf')), '300+'),
]

BUCKET_LABELS = [item[1] for item in FRAGMENT_RANGES]


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


# =============================================================================
# 1. FUSED C-ACCELERATED NUMBA KRUSKAL & O(N) ARI SWEEP (ZERO HEAP MALLOC)
# =============================================================================

@njit(fastmath=True)
def _kruskal_ari_sweep_numba(src, tgt, weights, thresholds_desc, n, gt_labels_clean, num_gt_clusters, sum_a, n_choose_2):
    """Numba C-kernel: Kruskal Union-Find with zero heap allocation in threshold loop."""
    num_edges = len(weights)
    num_t = len(thresholds_desc)

    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)

    # Pre-allocate temporary working buffers ONCE outside threshold loop
    pred_map = np.full(n, -1, dtype=np.int32)
    pred_labels = np.empty(n, dtype=np.int32)
    pred_counts = np.zeros(n, dtype=np.int64)
    contingency = np.zeros((num_gt_clusters, n), dtype=np.int64)

    ari_out = np.empty(num_t, dtype=np.float64)
    edge_idx = 0

    for t_idx in range(num_t):
        p_thresh = thresholds_desc[t_idx]

        # 1. Union edges meeting threshold
        while edge_idx < num_edges and weights[edge_idx] >= p_thresh:
            u = src[edge_idx]
            v = tgt[edge_idx]
            edge_idx += 1

            # Path compression U
            root_u = u
            while parent[root_u] != root_u:
                root_u = parent[root_u]
            curr = u
            while curr != root_u:
                nxt = parent[curr]
                parent[curr] = root_u
                curr = nxt

            # Path compression V
            root_v = v
            while parent[root_v] != root_v:
                root_v = parent[root_v]
            curr = v
            while curr != root_v:
                nxt = parent[curr]
                parent[curr] = root_v
                curr = nxt

            # Union by rank
            if root_u != root_v:
                if rank[root_u] < rank[root_v]:
                    root_u, root_v = root_v, root_u
                parent[root_v] = root_u
                if rank[root_u] == rank[root_v]:
                    rank[root_u] += 1

        # 2. Compress roots & map to contiguous predicted cluster IDs (0..num_pred-1)
        pred_map.fill(-1)
        num_pred_clusters = 0

        for i in range(n):
            root = i
            while parent[root] != root:
                root = parent[root]
            
            if pred_map[root] == -1:
                pred_map[root] = num_pred_clusters
                num_pred_clusters += 1
            pred_labels[i] = pred_map[root]

        # Reset active slice counts in-place
        for p in range(num_pred_clusters):
            pred_counts[p] = 0
            for g in range(num_gt_clusters):
                contingency[g, p] = 0

        # 3. Direct O(N) Contingency Accumulation
        for i in range(n):
            g = gt_labels_clean[i]
            p = pred_labels[i]
            pred_counts[p] += 1
            contingency[g, p] += 1

        # sum_b = sum(b_j choose 2)
        sum_b = 0.0
        for p in range(num_pred_clusters):
            cnt = pred_counts[p]
            if cnt > 1:
                sum_b += cnt * (cnt - 1) / 2.0

        # sum_nij = sum(n_ij choose 2)
        sum_nij = 0.0
        for g in range(num_gt_clusters):
            for p in range(num_pred_clusters):
                cnt = contingency[g, p]
                if cnt > 1:
                    sum_nij += cnt * (cnt - 1) / 2.0

        # 4. Compute Adjusted Rand Index (Hubert & Arabie 1985)
        expected_index = (sum_a * sum_b) / n_choose_2
        max_index = (sum_a + sum_b) / 2.0
        denominator = max_index - expected_index

        if denominator == 0.0:
            ari_out[t_idx] = 1.0 if sum_a == sum_b else 0.0
        else:
            ari_out[t_idx] = (sum_nij - expected_index) / denominator

    return ari_out


# =============================================================================
# 2. SPARSE KRUSKAL SINGLE-LINKAGE EVALUATOR
# =============================================================================

def _eval_single_sample_ari_sparse(probs_np, topk_local_np, gt_labels_np, thresholds, topk_eval=1):
    """Compute ARI across thresholds using Sparse Kruskal Union-Find.

    Args:
        probs_np: (N, K) array of edge probabilities
        topk_local_np: (N, K) array of local neighbor indices
        gt_labels_np: (N,) array of ground truth labels
        thresholds: array of threshold values to sweep
        topk_eval: number of top-K edges to use per point (1 = strongest only)
    """
    N = len(gt_labels_np)
    if N <= 1:
        return None

    n_choose_2 = N * (N - 1) / 2.0
    if n_choose_2 == 0:
        return None

    unique_gt, gt_labels_clean = np.unique(gt_labels_np, return_inverse=True)
    num_gt_clusters = len(unique_gt)
    gt_labels_clean = gt_labels_clean.astype(np.int32)

    _, gt_counts = np.unique(gt_labels_clean, return_counts=True)
    sum_a = float(np.sum(gt_counts * (gt_counts - 1) / 2.0))

    # Select top-K edges per point for ARI computation
    if topk_eval == 1:
        # Strongest edge only
        src = np.arange(N, dtype=np.int32)
        best_k = np.argmax(probs_np, axis=1)
        tgt = topk_local_np[np.arange(N), best_k].astype(np.int32)
        weights = probs_np[np.arange(N), best_k]
    else:
        # Top-K edges per point
        topk_idx = np.argsort(-probs_np, axis=1)[:, :topk_eval]
        src = np.repeat(np.arange(N, dtype=np.int32), topk_eval)
        tgt = topk_local_np[np.arange(N)[:, None], topk_idx].ravel().astype(np.int32)
        weights = probs_np[np.arange(N)[:, None], topk_idx].ravel()

    # Filter out self-loops and invalid targets
    valid = (src != tgt) & (tgt >= 0) & (tgt < N)

    # Thresholds descending
    desc_t_order = np.argsort(-thresholds)
    thresholds_desc = thresholds[desc_t_order]
    num_t = len(thresholds)

    if not np.any(valid):
        # Baseline ARI when no valid edges exist
        sum_b = 0.0
        expected_index = (sum_a * sum_b) / n_choose_2
        max_index = (sum_a + sum_b) / 2.0
        denom = max_index - expected_index
        base_ari = (0.0 - expected_index) / denom if denom != 0 else (1.0 if sum_a == sum_b else 0.0)
        return np.full(num_t, base_ari, dtype=np.float64)

    src_v = src[valid]
    tgt_v = tgt[valid]
    w_v = weights[valid]

    # Sort directed edges descending by weight
    order = np.argsort(-w_v)
    src_v = src_v[order]
    tgt_v = tgt_v[order]
    w_v = w_v[order]

    # Fused C-Kernel Kruskal + ARI sweep
    ari_desc = _kruskal_ari_sweep_numba(
        src_v, tgt_v, w_v, thresholds_desc, N,
        gt_labels_clean, num_gt_clusters, sum_a, n_choose_2
    )

    # Map back to original threshold order
    ari_results = np.empty(num_t, dtype=np.float64)
    for idx, orig_t_idx in enumerate(desc_t_order):
        ari_results[orig_t_idx] = ari_desc[idx]

    return ari_results


# =============================================================================
# 3. MAIN VALIDATION HARNESS
# =============================================================================

def validate(model, val_loader, criterion, device, max_batches=None, ari_topk=1, bucket_topk_map=None):
    """Run validation loop, sweeping thresholds to find best ARI via sparse single-linkage.
    
    Args:
        bucket_topk_map: Optional dict mapping bucket keys to topk values.
                        If provided, uses per-bucket topk for ARI evaluation.
    """
    model.eval()
    total_loss = 0.0
    total_scene_count = 0
    num_batches = 0
    total_samples = 0
    valid_sample_count_ari = 0

    thresholds_np = np.arange(0.2, 0.8, 0.02)
    num_thresholds = len(thresholds_np)
    ari_sums = np.zeros(num_thresholds, dtype=np.float64)

    buckets = {}
    for (low, high), range_key in FRAGMENT_RANGES:
        buckets[range_key] = {
            'ari_sums': np.zeros(num_thresholds, dtype=np.float64),
            'loss_sum': 0.0,
            'sample_count': 0,
            'valid_sample_count_ari': 0,
        }

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            embeddings = batch['embeddings'].to(device, non_blocking=True)
            transforms = batch['transforms'].to(device, non_blocking=True)
            asset_ids = batch['asset_ids'].to(device, non_blocking=True)
            scene_ids = batch.get('scene_ids', None)

            if scene_ids is not None:
                scene_ids = scene_ids.to(device, non_blocking=True)

            soup_stats = batch.get('soup_stats', [])

            with torch.amp.autocast(device_type=device_type):
                logits, indices, cand_mask = model(embeddings, transforms, scene_ids)
                scene_ids_t = scene_ids if scene_ids is not None else torch.zeros(len(asset_ids), dtype=torch.long, device=device)
                loss, loss_meta = criterion(logits, indices, asset_ids, cand_mask)

            unique_scene_indices = torch.unique(scene_ids_t)
            num_scenes_in_batch = len(unique_scene_indices)
            total_loss += loss.item() * num_scenes_in_batch
            total_scene_count += num_scenes_in_batch
            num_batches += 1

            asset_ids_np = asset_ids.cpu().numpy()
            logits_np = logits.cpu().numpy()
            indices_np = indices.cpu().numpy()
            mask_np = cand_mask.cpu().numpy()

            for batch_idx_local, scene_idx in enumerate(unique_scene_indices):
                scene_idx_int = int(scene_idx.item())
                idx_global_t = (scene_ids_t == scene_idx_int).nonzero(as_tuple=True)[0]
                N_s = len(idx_global_t)

                range_key = None
                # soup_stats is a list of dicts, one per sample in the batch
                if soup_stats and batch_idx_local < len(soup_stats):
                    actual_n = soup_stats[batch_idx_local].get('actual_n', 0)
                    range_key = _frag_to_range_key(actual_n)

                logits_s = logits_np[idx_global_t.cpu().numpy()]
                global_indices_s = indices_np[idx_global_t.cpu().numpy()]
                cand_mask_s = mask_np[idx_global_t.cpu().numpy()]

                # Apply candidate_mask to logits: padding positions get -inf
                logits_masked = torch.from_numpy(logits_s).clone()
                cmask_t = torch.from_numpy(cand_mask_s)
                logits_masked[cmask_t == 0] = float('-inf')
                probs_s_np = torch.sigmoid(logits_masked).cpu().numpy().astype(np.float64)
                scene_asset_ids_np = asset_ids_np[idx_global_t.cpu().numpy()]

                # Accelerated GPU Local Index Mapping
                if global_indices_s is not None:
                    local_indices_t = torch.clamp(torch.searchsorted(idx_global_t, torch.from_numpy(global_indices_s).to(device)), 0, max(0, N_s - 1))
                    local_indices_np = local_indices_t.cpu().numpy()
                else:
                    if probs_s_np.ndim == 2 and probs_s_np.shape[0] == probs_s_np.shape[1]:
                        local_indices_np = np.tile(np.arange(N_s, dtype=np.int32), (N_s, 1))
                    else:
                        K = probs_s_np.shape[1]
                        local_indices_np = np.tile(np.arange(min(K, N_s), dtype=np.int32), (N_s, 1))

                # Evaluate ARI
                if N_s <= 1:
                    continue

                topk_eval = bucket_topk_map.get(range_key, ari_topk) if bucket_topk_map else ari_topk
                b_ari = _eval_single_sample_ari_sparse(
                    probs_s_np, local_indices_np, scene_asset_ids_np,
                    thresholds_np, topk_eval=topk_eval
                )

                track_bucket = range_key is not None and range_key in buckets
                total_samples += 1

                if b_ari is not None:
                    valid_sample_count_ari += 1
                    ari_sums += b_ari
                    if track_bucket:
                        buckets[range_key]['ari_sums'] += b_ari
                        buckets[range_key]['valid_sample_count_ari'] += 1

                if track_bucket:
                    buckets[range_key]['loss_sum'] += loss.item()
                    buckets[range_key]['sample_count'] += 1

    # Finalize Per-Bucket Metrics
    avg_loss = total_loss / max(total_scene_count, 1)

    bucket_metrics = {}
    bucket_ari_curves = {}
    for range_key, bucket in buckets.items():
        if bucket['sample_count'] == 0:
            continue

        b_loss = bucket['loss_sum'] / bucket['sample_count']
        b_best_ari = -1.0
        b_best_threshold = 0.5
        if bucket['valid_sample_count_ari'] > 0:
            b_avg_ari = bucket['ari_sums'] / bucket['valid_sample_count_ari']
            b_best_t_idx = np.argmax(b_avg_ari)
            b_best_ari = float(b_avg_ari[b_best_t_idx])
            b_best_threshold = float(thresholds_np[b_best_t_idx])
            bucket_ari_curves[range_key] = b_avg_ari.tolist()
        else:
            bucket_ari_curves[range_key] = np.zeros(num_thresholds).tolist()

        bucket_metrics[range_key] = {
            'loss': round(b_loss, 4),
            'ari': round(b_best_ari, 4),
            'threshold': round(b_best_threshold, 2),
            'count': bucket['sample_count'],
        }

    return {
        'avg_loss': round(avg_loss, 4),
        'bucket_ari_curves': bucket_ari_curves,
        'bucket_metrics': bucket_metrics,
    }