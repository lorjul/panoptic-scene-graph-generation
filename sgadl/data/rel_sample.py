from typing import Optional
import torch


def idx_to_sbjobj(idx: torch.Tensor, num_instances: int):
    tmp_idx = idx + (idx // num_instances) + 1
    sbj = tmp_idx // num_instances
    obj = tmp_idx - (sbj * num_instances)
    return torch.stack((sbj, obj), dim=1)


def sbjobj_to_idx(sbj: torch.Tensor, obj: torch.Tensor, num_instances: int):
    """Self-relations are not allowed!"""
    assert (sbj != obj).all()
    return sbj * (num_instances - 1) + obj - (sbj < obj).long()


def sample_pairs(num_instances: int, pos_pairs: torch.Tensor, neg_ratio: Optional[float] = None):
    pos_idx = sbjobj_to_idx(
        sbj=pos_pairs[:, 0],
        obj=pos_pairs[:, 1],
        num_instances=num_instances,
    )
    is_negative = torch.ones(num_instances * (num_instances - 1), dtype=torch.bool)
    # select positives
    is_negative[pos_idx] = False

    # select negatives
    neg_ids = torch.nonzero(is_negative).flatten()

    # if neg_ratio is set, choose only a subset of negatives
    if neg_ratio is not None:
        num_neg = round(neg_ratio * len(pos_pairs))
        num_neg = min(num_neg, neg_ids.size(0))
        neg_ids = neg_ids[torch.randperm(neg_ids.size(0))[:num_neg]]

    all_ids = torch.cat((pos_idx, neg_ids))
    return idx_to_sbjobj(idx=all_ids, num_instances=num_instances)
