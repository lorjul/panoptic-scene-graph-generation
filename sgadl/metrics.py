from typing import Sequence, Optional
import torch
from sgadl.relparam import TargetThresholder


def _recall_k_counts(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    """A variant of the mean Recall@k metric for our model.
    To select the top-k predictions, our model can use the no-relation class output.
    This is different to other methods and therefore, we have to calculate the metric slightly different.
    """

    num_classes = output_list[0].size(1) - 1
    # count all annotations per class here (except no-relation class)
    gt_counts = torch.zeros(num_classes, dtype=torch.long)
    # count all hits per class (except no-relation class)
    hit_counts = torch.zeros(num_classes, dtype=torch.long)
    for y, x in zip(gt_list, output_list):
        gt_counts += y[:, 1:].sum(dim=0).long()

        # select the top-k predictions based on the no-relation class
        # low score means high likelihood of an annotated relation (torch.argsort is lowest first)
        ordering = torch.argsort(x[:, 0])[:k]
        # use argmax to select the highest score for the given relation (not ideal :/)
        topk_pred = x[ordering, 1:].argmax(dim=-1)
        # in the ground truth (which is one-hot-encoded), retrieve the annotations (either 0 or 1)
        is_correct = y[ordering, topk_pred + 1].long()
        # increase the number of hits per class (topk_pred chooses the predicate class index, is_correct if it's a hit or a miss)
        hit_counts.scatter_add_(0, topk_pred, is_correct)

    return gt_counts, hit_counts


def mean_recall_k(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    gt_counts, hit_counts = _recall_k_counts(
        k=k, gt_list=gt_list, output_list=output_list
    )
    per_class_recall = hit_counts / gt_counts
    assert not torch.isnan(per_class_recall).any(), per_class_recall
    return per_class_recall.mean().item()


def recall_k(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    gt_counts, hit_counts = _recall_k_counts(
        k=k, gt_list=gt_list, output_list=output_list
    )
    return hit_counts.sum() / gt_counts.sum()


def _nogc_recall_k_counts(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    """A variant of the mean Recall@k with no graph constraint for out model"""
    num_classes = output_list[0].size(1) - 1
    gt_counts = torch.zeros(num_classes, dtype=torch.long)
    hit_counts = torch.zeros(num_classes, dtype=torch.long)
    for y, x in zip(gt_list, output_list):
        gt_counts += y[:, 1:].sum(dim=0).long()

        # TODO: how do I obtain a good predicate score here?
        fg_score = (1 - x[:, 0])[:, None]
        modx = fg_score * x[:, 1:]

        ordering = modx.flatten().argsort()[-k:]

        class_ids = torch.tile(torch.arange(len(gt_counts)), (len(y),))
        is_correct = y[:, 1:].flatten()[ordering].long()
        hit_counts.scatter_add_(0, class_ids[ordering], is_correct)

    return gt_counts, hit_counts


def nogc_recall_k(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    gt_counts, hit_counts = _nogc_recall_k_counts(
        k=k, gt_list=gt_list, output_list=output_list
    )
    return hit_counts.sum() / gt_counts.sum()


def mean_nogc_recall_k(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    gt_counts, hit_counts = _nogc_recall_k_counts(
        k=k, gt_list=gt_list, output_list=output_list
    )
    per_class_recall = hit_counts / gt_counts
    assert not torch.isnan(per_class_recall).any(), per_class_recall
    return per_class_recall.mean().item()


def _per_class(gts: torch.Tensor, hits: torch.Tensor, nan_ok=False):
    per_class = hits / gts
    assert nan_ok or not torch.isnan(per_class).any(), per_class
    return per_class


def build_rel_metrics_dict(
    rel_names: Sequence[str],
    rel_thresher: Optional[TargetThresholder],
    gt_list: Sequence[torch.Tensor],
    output_list: Sequence[torch.Tensor],
    param_gt_list: Sequence[torch.Tensor],
    nan_ok=False,
):
    if rel_thresher is None:
        thresh_gt_list = gt_list
    else:
        thresh_gt_list = []
        for gt, param in zip(gt_list, param_gt_list):
            thresh_gt = rel_thresher(target_class=gt, target_param=param)
            # Set the ignore class as a negative label (=0).
            # This should be fine because R@k is TP/(TP+FN)
            # Hence the ignore class has no direct influence on R@k

            # TODO: we could introduce a more involved handling of the ignore class:
            # If the model selects an ignore class, it should get an additional attempt, so k+=1.
            # However, this is non-trivial to implement because often, only one label of the relation will be set to ignore.
            # Until we find a better approach, just handle the ignore class as a negative label.
            thresh_gt[thresh_gt == -1] = 0

            thresh_gt_list.append(thresh_gt)

    r1000_g, r1000_o = _recall_k_counts(
        k=1000, gt_list=thresh_gt_list, output_list=output_list
    )
    r50_g, r50_o = _recall_k_counts(
        k=50, gt_list=thresh_gt_list, output_list=output_list
    )
    n1000_g, n1000_o = _nogc_recall_k_counts(
        k=1000, gt_list=thresh_gt_list, output_list=output_list
    )
    n50_g, n50_o = _nogc_recall_k_counts(
        k=50, gt_list=thresh_gt_list, output_list=output_list
    )

    recall1000_classes = _per_class(r1000_g, r1000_o, nan_ok=nan_ok)
    recall50_classes = _per_class(r50_g, r50_o, nan_ok=nan_ok)
    nogc1000_classes = _per_class(n1000_g, n1000_o, nan_ok=nan_ok)
    nogc50_classes = _per_class(n50_g, n50_o, nan_ok=nan_ok)

    assert len(rel_names) == len(recall1000_classes)

    metrics = {
        "rel_recall/1000": r1000_o.sum() / r1000_g.sum(),
        "rel_recall/50": r50_o.sum() / r50_g.sum(),
        "rel_mean_recall/1000": recall1000_classes.nanmean(),
        "rel_mean_recall/50": recall50_classes.nanmean(),
        "rel_nogc_recall/1000": n1000_o.sum() / n1000_g.sum(),
        "rel_nogc_recall/50": n50_o.sum() / n50_g.sum(),
        "rel_mean_nogc_recall/1000": nogc1000_classes.nanmean(),
        "rel_mean_nogc_recall/50": nogc50_classes.nanmean(),
    }

    for name, v in zip(rel_names, recall1000_classes):
        metrics[f"rel_class_recall/1000-{name}"] = v
    for name, v in zip(rel_names, nogc1000_classes):
        metrics[f"rel_class_nogc_recall/1000-{name}"] = v

    return {k: v.item() for k, v in metrics.items()}
