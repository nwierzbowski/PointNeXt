"""Validation functions for peeler training."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


def _gt_connected_components(Y, mask):
    """Extract connected components from ground truth NxN same-asset matrix.

    Vectorized version using iterative label propagation.

    Args:
        Y: (B, N, N) - ground truth same-asset matrix (bool or 0/1)
        mask: (B, N) - 1 for real fragments, 0 for padding

    Returns:
        labels: (B, N) - cluster assignment for each fragment (-1 for padding)
    """
    B, N, _ = Y.shape
    device = Y.device

    # Y should be boolean or thresholded
    adj = Y > 0.5
    pad_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).bool()
    valid_adj = adj & pad_mask

    indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    labels = indices.clone()
    labels = labels.masked_fill(~mask.bool(), -1)

    for _ in range(50):
        neighbor_labels = labels.unsqueeze(1).expand(-1, N, -1)
        neighbor_labels = neighbor_labels.masked_fill(~valid_adj, N)
        new_labels = neighbor_labels.min(dim=2).values
        if torch.equal(new_labels, labels):
            break
        labels = new_labels

    return labels


def _ari_nmi_batch(labels_true, labels_pred, mask):
    """Compute ARI and NMI for a batch of label assignments.

    Vectorized operations using numpy for speed.

    Args:
        labels_true: (B, N) - ground truth cluster labels
        labels_pred: (B, N) - predicted cluster labels
        mask: (B, N) - 1 for valid fragments

    Returns:
        ari_sum: sum of ARI scores
        nmi_sum: sum of NMI scores
        batch_count: number of samples with >= 2 valid fragments
    """
    ari_sum = 0.0
    nmi_sum = 0.0
    batch_count = 0

    B, N = labels_true.shape
    for b in range(B):
        gt = labels_true[b].cpu().numpy()
        pred = labels_pred[b].cpu().numpy()
        valid = mask[b].cpu().numpy() > 0.5

        gt_valid = gt[valid]
        pred_valid = pred[valid]

        if len(gt_valid) < 2:
            continue

        # Build contingency table using numpy histogram2d
        unique_gt = np.unique(gt_valid)
        unique_pred = np.unique(pred_valid)
        contingency = np.zeros((len(unique_gt), len(unique_pred)), dtype=np.int64)

        gt_idx = np.searchsorted(unique_gt, gt_valid)
        pred_idx = np.searchsorted(unique_pred, pred_valid)

        for i in range(len(gt_valid)):
            contingency[gt_idx[i], pred_idx[i]] += 1

        # ARI
        n = len(gt_valid)
        sum_comb_ij = (contingency * (contingency - 1)).sum() / 2
        a = contingency.sum(axis=1)
        b = contingency.sum(axis=0)
        sum_comb_a = (a * (a - 1)).sum() / 2
        sum_comb_b = (b * (b - 1)).sum() / 2
        n_comb = n * (n - 1) / 2
        expected = sum_comb_a * sum_comb_b / n_comb if n_comb > 0 else 0
        max_index = (sum_comb_a + sum_comb_b) / 2
        denom = max_index - expected

        if denom == 0:
            ari = 1.0 if sum_comb_ij == expected else 0.0
        else:
            ari = (sum_comb_ij - expected) / denom

        # NMI
        row_sum = contingency.sum(axis=1)
        col_sum = contingency.sum(axis=0)
        total = contingency.sum()

        mi = 0.0
        for i in range(len(unique_gt)):
            for j in range(len(unique_pred)):
                if contingency[i, j] > 0:
                    p_ij = contingency[i, j] / total
                    p_i = row_sum[i] / total
                    p_j = col_sum[j] / total
                    mi += p_ij * math.log(p_ij / (p_i * p_j))

        def entropy(m):
            h = 0.0
            for v in m:
                if v > 0:
                    p = v / m.sum()
                    h -= p * math.log(p)
            return h

        h_true = entropy(row_sum)
        h_pred = entropy(col_sum)

        if h_true == 0 or h_pred == 0:
            nmi = 1.0 if len(unique_gt) == len(unique_pred) else 0.0
        else:
            nmi = mi / math.sqrt(h_true * h_pred)

        ari_sum += ari
        nmi_sum += nmi
        batch_count += 1

    return ari_sum, nmi_sum, batch_count


def _f1_score_batch(Y_true, Y_pred, mask):
    """Compute F1, precision, recall for pairwise affinity prediction.

    Args:
        Y_true: (B, N, N) - ground truth same-asset matrix (0/1)
        Y_pred: (B, N, N) - predicted affinity matrix (0/1, thresholded)
        mask: (B, N) - 1 for valid fragments

    Returns:
        tp: true positives count
        fp: false positives count
        fn: false negatives count
    """
    B, N, _ = Y_true.shape
    tp = 0.0
    fp = 0.0
    fn = 0.0

    valid = mask.unsqueeze(1) * mask.unsqueeze(2)
    diag_mask = 1.0 - torch.eye(N, device=Y_true.device).unsqueeze(0)
    eval_mask = valid * diag_mask

    for b in range(B):
        true_b = (Y_true[b] > 0.5) & (eval_mask[b] > 0.5)
        pred_b = (Y_pred[b] > 0.5) & (eval_mask[b] > 0.5)
        tp += float((true_b & pred_b).sum().item())
        fp += float((~true_b & pred_b).sum().item())
        fn += float((true_b & ~pred_b).sum().item())

    return tp, fp, fn


def _dbscan_embeddings(embeddings, mask, eps=0.5, min_samples=2):
    """Run DBSCAN on refined embeddings per batch item using sklearn.

    Matches the Rust inference DBSCAN parameters (eps=0.5, min_samples=2).

    Args:
        embeddings: (B, N, D) - refined embeddings (on GPU)
        mask: (B, N) - 1 for valid fragments, 0 for padding
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        labels: (B, N) - cluster assignment for each fragment (-1 for noise/padding)
    """
    B, N, D = embeddings.shape
    labels = torch.full((B, N), -1, dtype=torch.long, device=embeddings.device)

    # Move to CPU for sklearn
    emb_np = embeddings.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    for b in range(B):
        valid = mask_np[b] > 0.5
        valid_indices = np.where(valid)[0]
        if len(valid_indices) < min_samples:
            continue

        points = emb_np[b, valid_indices]  # (num_valid, D)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        cluster_labels = db.labels_  # -1 for noise, 0+ for clusters

        # Map back to full N-sized array
        for idx, local_idx in enumerate(valid_indices):
            labels[b, local_idx] = torch.tensor(cluster_labels[idx], dtype=torch.long, device=embeddings.device)

    return labels


def validate(model, val_loader, criterion, device, scaler, epoch, num_epochs):
    """Run validation loop and return loss, ARI, and NMI."""
    model.eval()
    total_loss = 0.0
    ari_sum = 0.0
    nmi_sum = 0.0
    batch_count = 0
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0

    with torch.no_grad():
        for (batch_idx, batch) in enumerate(val_loader):
            embeddings = batch['embeddings'].to(device)
            transforms = batch['transforms'].to(device)
            mask = batch['mask'].to(device)
            Y = batch['Y'].to(device).float()
            asset_ids = batch['asset_ids'].to(device, dtype=torch.long)

            model_name = type(model).__name__
            if model_name == 'PurelyRelationalPeeler':
                affinity_logits = model(transforms, mask)
                loss, loss_dict = criterion(affinity_logits, Y, mask, epoch, num_epochs)

                # F1 on pairwise affinity
                pred_adj = (torch.sigmoid(affinity_logits) > 0.5).float()
                tp, fp, fn = _f1_score_batch(Y, pred_adj, mask)
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn

                # Connected components for clustering metrics
                pred_labels = _gt_connected_components(pred_adj, mask)
            else:
                refined_emb = model(embeddings, transforms, mask)
                loss, loss_dict = criterion(refined_emb, asset_ids, mask, epoch, num_epochs)

                # DBSCAN evaluation on refined embeddings (matches Rust inference)
                pred_labels = _dbscan_embeddings(refined_emb, mask, eps=0.5, min_samples=2)

            gt_labels = _gt_connected_components(Y, mask)

            # Compute ARI and NMI for the batch
            ari, nmi, cnt = _ari_nmi_batch(gt_labels, pred_labels, mask)
            ari_sum += ari
            nmi_sum += nmi
            batch_count += cnt

            total_loss += loss.item()

    num_batches = max(len(val_loader), 1)
    avg_loss = total_loss / num_batches
    avg_ari = ari_sum / batch_count if batch_count > 0 else 0.0
    avg_nmi = nmi_sum / batch_count if batch_count > 0 else 0.0

    # Compute F1 from accumulators
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return avg_loss, avg_ari, avg_nmi, f1
