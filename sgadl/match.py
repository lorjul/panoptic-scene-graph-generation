# match predicted segmentatino mask with ground truth and return the ground truth id
import numpy as np


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def box_area(boxes: np.ndarray):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def match_boxes(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    min_thresh: float = 0.5,
):
    assert gt_boxes.shape[1] == 4
    assert pred_boxes.shape[1] == 4
    assert min_thresh > 0, min_thresh
    num_pred = len(pred_boxes)

    ious = box_iou(pred_boxes, gt_boxes)

    label_match = pred_labels[:, None] == gt_labels[None]
    assert label_match.shape == ious.shape

    # find the ground truth that overlaps most and has the correct label
    # if overlap is highest, but label does not match, use the second highest overlap
    ious[~label_match] = 0

    # all masks that could not be matched get a -1
    assigned_gt = ious.argmax(-1)
    iou_vals = ious[np.arange(num_pred), assigned_gt]
    assigned_gt[iou_vals < min_thresh] = -1

    return assigned_gt


def match_masks(
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    min_thresh: float = 0.5,
):
    assert len(gt_masks.shape) == 3
    assert len(pred_masks.shape) == 3
    assert gt_masks.shape[1:] == pred_masks.shape[1:]

    label_match = pred_labels[:, None] == gt_labels[None]
    ious = np.zeros((len(pred_masks), len(gt_masks)))
    num_pred = len(ious)
    assert label_match.shape == ious.shape

    for pi, pm in enumerate(pred_masks):
        for gi, gm in enumerate(gt_masks):
            # only calculate IoU if labels match
            if label_match[pi, gi]:
                ious[pi, gi] = mask_iou(pm, gm)

    # all masks that could not be matched get a -1
    assigned_gt = ious.argmax(-1)
    iou_vals = ious[np.arange(num_pred), assigned_gt]
    assigned_gt[iou_vals < min_thresh] = -1

    return assigned_gt


def match_instances(
    gt_instances: np.ndarray,
    gt_labels: np.ndarray,
    pred_instances: np.ndarray,
    pred_labels: np.ndarray,
) -> np.ndarray:
    if len(gt_instances.shape) == 2 and gt_instances.shape[1] == 4:
        return match_boxes(
            gt_boxes=gt_instances,
            pred_boxes=pred_instances,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            min_thresh=0.5,
        )
    else:
        return match_masks(
            gt_masks=gt_instances,
            pred_masks=pred_instances,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            min_thresh=0.5,
        )


def remap_triplets(inst_pred2gt: np.ndarray, pred_triplets: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            inst_pred2gt[pred_triplets[:, 0]],
            inst_pred2gt[pred_triplets[:, 1]],
            pred_triplets[:, 2],
        ),
        axis=1,
    )
