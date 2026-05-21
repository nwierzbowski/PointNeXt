"""Peeler inference - Iterative peeling loop.

Takes a scene of N fragments and iteratively peels assets:
    1. Run transformer → contextual embeddings
    2. Anchor head → pick best seed
    3. Relation head → group members
    4. Remove grouped fragments
    5. Repeat until anchor score < threshold
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional


@torch.no_grad()
def peel_asset(
    model: nn.Module,
    embeddings: torch.Tensor,
    transforms: torch.Tensor,
    device: torch.device,
    membership_threshold: float = 0.5,
) -> Optional[Dict]:
    """Peel a single asset from the soup.

    Args:
        model: trained Peeler model (eval mode)
        embeddings: (N, 256) - fragment embeddings
        transforms: (N, 16) - fragment transforms
        device: torch device
        membership_threshold: minimum membership prob to include in asset

    Returns:
        dict with member_indices, anchor_idx, anchor_score, membership_probs
        or None if no fragments remain
    """
    if len(embeddings) == 0:
        return None

    # Add batch dimension
    emb_batch = embeddings.unsqueeze(0).to(device)  # (1, N, 256)
    trans_batch = transforms.unsqueeze(0).to(device)  # (1, N, 16)

    anchor_probs, membership_logits, _, seed_idx = model(emb_batch, trans_batch)
    anchor_probs = anchor_probs[0]  # (N,)
    membership_probs = torch.sigmoid(membership_logits[0])  # (N,)
    seed_idx = seed_idx.item()  # scalar

    # Select members
    member_mask = membership_probs >= membership_threshold
    member_indices = torch.where(member_mask)[0].cpu().tolist()

    if not member_indices:
        return None

    return {
        'member_indices': member_indices,
        'anchor_idx': seed_idx,
        'anchor_score': anchor_probs[seed_idx].item(),
        'membership_probs': membership_probs.cpu(),
    }


@torch.no_grad()
def peel_scene(
    model: nn.Module,
    all_embeddings: torch.Tensor,
    all_transforms: torch.Tensor,
    device: torch.device,
    membership_threshold: float = 0.5,
    max_iterations: int = 50,
) -> List[Dict]:
    """Iteratively peel all assets from a scene.

    Args:
        model: trained Peeler model (eval mode)
        all_embeddings: (N, 256) - all fragment embeddings
        all_transforms: (N, 16) - all fragment transforms
        device: torch device
        membership_threshold: minimum membership prob to include
        max_iterations: maximum number of peeling iterations

    Returns:
        list of dicts, one per peeled asset
    """
    model.eval()

    remaining_idx = list(range(len(all_embeddings)))
    assets = []

    for iteration in range(max_iterations):
        if not remaining_idx:
            break

        remaining_tensor = torch.tensor(remaining_idx, dtype=torch.long)
        embeddings = all_embeddings[remaining_tensor]
        transforms = all_transforms[remaining_tensor]

        result = peel_asset(
            model, embeddings, transforms, device, membership_threshold,
        )

        if result is None:
            break

        original_members = [remaining_idx[i] for i in result['member_indices']]
        original_anchor = remaining_idx[result['anchor_idx']]

        assets.append({
            'member_indices': original_members,
            'anchor_idx': original_anchor,
            'anchor_score': result['anchor_score'],
            'iteration': iteration,
        })

        peeled_set = set(result['member_indices'])
        remaining_idx = [
            i for i, orig in enumerate(remaining_idx)
            if i not in peeled_set
        ]

    return assets
