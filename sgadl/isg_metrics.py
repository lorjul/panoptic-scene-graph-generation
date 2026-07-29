# metrics for ISG
from typing import Sequence, Optional
import torch
from sklearn.metrics import average_precision_score
from sgadl.relparam import (
    names_to_rel_types,
    relnames_to_cols,
    RELTYPE_ANGLE,
    RELTYPE_DISTANCE,
    TargetThresholder,
)


def angle_error(gt: torch.Tensor, gt_class: torch.Tensor, out: torch.Tensor):
    assert gt_class.dtype == torch.bool, gt_class.dtype
    # convert model output and ground truth to degrees
    # calculate absolute error in degrees
    rad = (torch.acos(gt.clamp(0, 1)) - torch.acos(out.clamp(0, 1))).abs()
    rad[~gt_class] = torch.nan
    return torch.rad2deg(rad).nanmean(0)


def dist_error(gt: torch.Tensor, gt_class: torch.Tensor, out: torch.Tensor):
    assert gt_class.dtype == torch.bool, gt_class.dtype
    dist = (gt - out).abs()
    dist[~gt_class] = torch.nan
    return dist.nanmean(0)


def fine_dist_error(gt: torch.Tensor, out: torch.Tensor):
    errs = {}
    errs["next_to"] = (gt[gt <= 1] - out[gt <= 1]).abs().mean()
    errs["not_next_to"] = (gt[gt > 1] - out[gt > 1]).abs().mean()

    # more fine-grained
    s = gt <= 0.3
    errs["proximal"] = (gt[s] - out[s]).abs().mean()
    s = (gt > 0.3) & (gt <= 1)
    errs["adjacent"] = (gt[s] - out[s]).abs().mean()
    s = (gt > 1) & (gt <= 3)
    errs["close"] = (gt[s] - out[s]).abs().mean()
    s = gt > 3
    errs["away"] = (gt[s] - out[s]).abs().mean()
    return errs


def build_isg_metrics_dict(
    rel_names: Sequence[str],
    rel_thresher: Optional[TargetThresholder],
    gt_list: Sequence[torch.Tensor],
    output_list: Sequence[torch.Tensor],
    param_gt_list: Sequence[torch.Tensor],
    param_output_list: Optional[Sequence[torch.Tensor]],
):
    # Concatenate all relations to one big tensor.
    # Alternatively, we could estimate the scores per image.
    # However, in practise this doesn't change much.
    gt = torch.cat(gt_list)
    output = torch.cat(output_list)
    param_gt = torch.cat(param_gt_list)

    if rel_thresher is None:
        thresh_gt = gt
    else:
        # we need to threshold the angles of the ground truth to get the actual target values
        thresh_gt = rel_thresher(target_class=gt, target_param=param_gt)

    aps = []
    for i in range(thresh_gt.shape[1]):
        sel = thresh_gt[:, i] != -1
        if sel.any():
            aps.append(average_precision_score(thresh_gt[sel, i], output[sel, i]))
        else:
            print("WARNING: Empty selection during AP calculation")
            aps.append(torch.nan)
    aps = torch.tensor(aps)

    metrics = {
        # ignore the norel class here
        "isg_mAP": aps[1:].nanmean(),
    }

    for name, v in zip(rel_names, aps):
        metrics[f"isg_AP/{name}"] = v

    # now, parameter metrics
    if param_output_list is not None:
        param_out = torch.cat(param_output_list)

        rel_types = names_to_rel_types(rel_names)
        is_angle, is_distance = relnames_to_cols(rel_names)

        aerr = angle_error(gt=param_gt, gt_class=gt.bool() & is_angle, out=param_out)
        derr = dist_error(gt=param_gt, gt_class=gt.bool() & is_distance, out=param_out)

        for name, rt, angle, distance in zip(rel_names, rel_types, aerr, derr):
            if rt == RELTYPE_ANGLE:
                metrics[f"angle/{name}"] = angle
            elif rt == RELTYPE_DISTANCE:
                metrics[f"dist/{name}"] = distance
            # else ignore

        metrics["angle/avg"] = aerr.nanmean()
        metrics["dist/avg"] = derr.nanmean()

        # more fine grained distances
        assert is_distance.sum() == 1
        dist_col = is_distance.nonzero()[0, 0]
        for k, v in fine_dist_error(
            gt=param_gt[:, dist_col], out=param_out[:, dist_col]
        ).items():
            metrics[f"dist/{k}"] = v

    return {k: float(v) for k, v in metrics.items()}
