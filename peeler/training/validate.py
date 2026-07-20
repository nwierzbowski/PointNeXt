"""Validation functions for peeler training."""
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components

def _get_linkage_matrix(affinity_matrix):
    """Compute the hierarchical linkage matrix Z.

    Args:
        affinity_matrix: (N, N) numpy array - sigmoid-activated affinities [0, 1]

    Returns:
        Z: linkage matrix, or None if N <= 1
    """
    N = affinity_matrix.shape[0]
    if N <= 1:
        return None

    # 2. Convert the symmetric affinity to a distance matrix
    distance_matrix = 1.0 - affinity_matrix

    # 3. Extract upper triangle and compute linkage
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method='average')
    return Z

def _get_valid_elements_for_f1(Y_true, affinity_probs, mask):
    """Extract valid elements for rapid multi-threshold F1 evaluation.

    Args:
        Y_true: (B, N, N) - ground truth same-asset matrix (0/1)
        affinity_probs: (B, N, N) - predicted affinity probabilities (sigmoid)
        mask: (B, N) - 1 for valid fragments

    Returns:
        y_true_valid: 1D boolean tensor of ground truth
        y_pred_probs_valid: 1D float tensor of predicted probabilities
    """
    N = Y_true.shape[1]
    valid = mask.unsqueeze(1) * mask.unsqueeze(2)
    diag_mask = 1.0 - torch.eye(N, device=Y_true.device).unsqueeze(0)
    eval_mask = (valid * diag_mask) > 0.5

    y_true_valid = Y_true[eval_mask] > 0.5
    y_pred_probs_valid = affinity_probs[eval_mask]
    return y_true_valid, y_pred_probs_valid

def _compute_f1_stats_from_valid(y_true_valid, y_pred_probs_valid, t):
    """Compute true positives, false positives, and false negatives from 1D tensors.

    Args:
        y_true_valid: 1D boolean tensor of ground truth
        y_pred_probs_valid: 1D float tensor of predicted probabilities
        t: float - threshold value

    Returns:
        tp, fp, fn: float counts
    """
    y_pred_valid = y_pred_probs_valid > t
    tp = (y_true_valid & y_pred_valid).sum().item()
    fp = (~y_true_valid & y_pred_valid).sum().item()
    fn = (y_true_valid & ~y_pred_valid).sum().item()
    return float(tp), float(fp), float(fn)

def validate(model, val_loader, criterion, device, scaler, epoch, num_epochs, 
             threshold):
    """Run validation loop, sweeping over thresholds to find the best ARI and F1.

    Returns:
        avg_loss: float
        best_ari: float
        best_ari_threshold: float
        best_f1: float
        best_f1_threshold: float
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0

    thresholds = np.arange(0.0, 1.0, 0.01)

    # Initialize tracking structures for each threshold
    ari_sums = {t: 0.0 for t in thresholds}
    tp_sums = {t: 0.0 for t in thresholds}
    fp_sums = {t: 0.0 for t in thresholds}
    fn_sums = {t: 0.0 for t in thresholds}

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embeddings'].to(device)
            transforms = batch['transforms'].to(device)
            mask = batch['mask'].to(device)
            Y = batch['Y'].to(device).float()

            asset_ids = batch.get('asset_ids')
            asset_ids_np = asset_ids.cpu().numpy()

            # Forward pass within autocast
            with torch.amp.autocast(device_type=device_type):
                affinity_logits = model(embeddings, transforms, mask)
                loss, _ = criterion(affinity_logits, Y, mask)

            total_loss += loss.item()

            # Compute sigmoid probabilities once
            affinity_probs = torch.sigmoid(affinity_logits)

            # Extract valid elements once for rapid 1D F1 sweeps
            y_true_valid, y_pred_probs_valid = _get_valid_elements_for_f1(Y, affinity_probs, mask)

            for t in thresholds:
                tp, fp, fn = _compute_f1_stats_from_valid(y_true_valid, y_pred_probs_valid, t)
                tp_sums[t] += tp
                fp_sums[t] += fp
                fn_sums[t] += fn

            # Bring only raw probabilities to CPU for clustering
            soft_A = affinity_probs.cpu().numpy()
            mask_np = mask.cpu().numpy()
            Y_np = Y.cpu().numpy()

            B = soft_A.shape[0]
            for b in range(B):
                active_indices = np.where(mask_np[b] > 0.5)[0]

                b_affinities = soft_A[b][active_indices][:, active_indices]

                # Extract ground-truth labels
                if asset_ids_np is not None:
                    gt_labels = asset_ids_np[b][active_indices]
                else:
                    b_gt_matrix = Y_np[b][active_indices][:, active_indices]
                    _, gt_labels = connected_components(csgraph=b_gt_matrix, directed=False)

                # BUILD the tree once for this sample
                Z = _get_linkage_matrix(b_affinities)
                if Z is None:
                    continue

                # CUT the tree and compute ARI for each threshold
                for t in thresholds:
                    pred_labels = fcluster(Z, t=t, criterion='distance')
                    ari_sums[t] += adjusted_rand_score(gt_labels, pred_labels)
                
                batch_count += 1

    # Finalize calculations
    num_batches = max(len(val_loader), 1)
    avg_loss = total_loss / num_batches

    # Find the best ARI and its corresponding threshold
    best_ari = -1.0
    best_ari_threshold = 0.5
    
    # Find the best F1 and its corresponding threshold
    best_f1 = -1.0
    best_f1_threshold = 0.5

    for t in thresholds:
        # ARI calculation
        avg_ari_t = ari_sums[t] / batch_count if batch_count > 0 else 0.0
        if avg_ari_t > best_ari:
            best_ari = avg_ari_t
            best_ari_threshold = t

        # F1 calculation
        precision_t = tp_sums[t] / (tp_sums[t] + fp_sums[t]) if (tp_sums[t] + fp_sums[t]) > 0 else 0.0
        recall_t = tp_sums[t] / (tp_sums[t] + fn_sums[t]) if (tp_sums[t] + fn_sums[t]) > 0 else 0.0
        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0.0
        
        if f1_t > best_f1:
            best_f1 = f1_t
            best_f1_threshold = t

    return avg_loss, best_ari, best_f1, best_ari_threshold, best_f1_threshold