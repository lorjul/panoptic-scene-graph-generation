from typing import Literal, Optional
import torch
from torch import Tensor
import torch.nn as nn
from sgadl.relparam import relnames_to_cols, TargetThresholder


def get_node_criterion(num_samples: torch.Tensor):
    valid = num_samples > 0
    if not valid.all():
        print("WARNING: This dataset has node labels without samples.")

    num_samples = num_samples.clone()
    num_samples[~valid] = 1

    inv = 1 / num_samples.float()
    class_weights = inv / inv[valid].mean()
    class_weights[~valid] = 0
    return nn.CrossEntropyLoss(weight=class_weights)


def get_multi_rel_criterion(pos_neg_ratios: torch.Tensor):
    pos_weights = 1 / pos_neg_ratios
    assert (pos_weights >= 0).all()
    assert not pos_weights.isinf().any()
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


def get_thresh_criterion(
    rel_thresher: Optional[TargetThresholder], pos_neg_ratios: torch.Tensor
):
    pos_weights = 1 / pos_neg_ratios
    assert (pos_weights >= 0).all()
    assert not pos_weights.isinf().any()
    if rel_thresher is None:
        return NoThreshWrapper(pos_weight=pos_weights)
    else:
        return ThreshBCEWithLogitsLoss(
            rel_thresher=rel_thresher, pos_weight=pos_weights
        )


class ThreshBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        rel_thresher: TargetThresholder,
        weight=None,
        pos_weight=None,
    ):
        super().__init__(weight=weight, reduction="none", pos_weight=pos_weight)
        self.thresher = rel_thresher

    def forward(
        self,
        input_class: torch.Tensor,
        target_class: torch.Tensor,
        target_param: torch.Tensor,
    ) -> torch.Tensor:
        thresh_target = self.thresher(
            target_class=target_class, target_param=target_param
        )

        # mask the ignore labels
        mask = thresh_target >= 0
        raw_loss = super().forward(input_class, thresh_target.float()) * mask
        return raw_loss.sum() / mask.sum()


class BCEWithLogitsIgnoreNegative(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        pos_weight: Optional[Tensor] = None,
        collect_stats: Optional[int] = None,
    ) -> None:
        """
        :param collect_stats: Set this parameter to the number of target classes to count the
        number of positives/negatives until .reset_stats() is called.
        """
        assert reduction in ("none", "mean", "sum")
        super().__init__(weight=weight, reduction="none", pos_weight=pos_weight)
        self.my_reduction = reduction
        if collect_stats is not None:
            self.pos_count_stats = torch.zeros(collect_stats, dtype=torch.long)
            self.neg_count_stats = torch.zeros(collect_stats, dtype=torch.long)
        else:
            self.pos_count_stats = None
            self.neg_count_stats = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # negative values indicate that they should be skipped
        if self.pos_count_stats is not None:
            assert target.size(1) == self.pos_count_stats.size(0)
            self.pos_count_stats += (target == 1).sum(0)
            self.neg_count_stats += (target == 0).sum(0)
        mask = target >= 0
        loss = super().forward(input, target.float()) * mask
        if self.my_reduction == "none":
            return loss
        if self.my_reduction == "sum":
            return loss.sum()
        if self.my_reduction == "mean":
            return loss.sum() / mask.sum()
        raise RuntimeError()

    def reset_stats(self):
        if self.pos_count_stats is not None:
            self.pos_count_stats[:] = 0
            self.neg_count_stats[:] = 0


class NoThreshWrapper(BCEWithLogitsIgnoreNegative):
    def forward(
        self,
        input_class: torch.Tensor,
        target_class: torch.Tensor,
        target_param: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(input_class, target_class)


def prel_angle_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert target.min() >= 0 and target.max() <= 1
    # all inputs are expected to be angles; without sigmoid, so ranging in (-inf, +inf)
    return torch.nn.functional.mse_loss(input.sigmoid(), target)


def prel_distance_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert (target >= 0).all()
    raw_loss = torch.nn.functional.smooth_l1_loss(
        input, target, beta=1.0, reduction="none"
    )
    # scale the raw loss with the distance (closer is more important)
    scaled_loss = raw_loss / (target + 1)
    # condense to a single value
    return scaled_loss.mean()


class ParamRelLoss(nn.Module):
    def __init__(self, rel_names, angle_weight: float, distance_weight: float):
        super().__init__()
        self.angle_cols, self.distance_cols = relnames_to_cols(rel_names)
        self.angle_weight = angle_weight
        self.distance_weight = distance_weight

    def forward(
        self,
        input_param: torch.Tensor,
        target_param: torch.Tensor,
        target_class: torch.Tensor,
    ):
        # only train parameters on relations where a ground truth annotation exists
        angle_mask = torch.zeros(
            target_class.shape, dtype=torch.bool, device=target_class.device
        )
        angle_mask[:, self.angle_cols] = True
        angle_mask = angle_mask & target_class.bool()
        if angle_mask.any():
            angle_scale = angle_mask[:, self.angle_cols].float().mean()
            # scale the loss by the inverse ratio
            # imagine two scenarios:
            # 1) there are 8 out of 8 angles present
            # 2) there are 4 out of 8 angles present
            # if the model would predict all angles with the exact same error,
            # we also want that the loss stays the same
            # Therefore:
            #   1) loss_1 = l / (8/8) = l
            #   2) loss_2 = l / (4/8) = 2*l
            angle_loss = (
                prel_angle_loss(input_param[angle_mask], target_param[angle_mask])
                / angle_scale
            )
        else:
            angle_loss = torch.tensor(0.0)

        # only train parameters on relations where a ground truth annotation exists
        distance_mask = torch.zeros(
            target_class.shape, dtype=torch.bool, device=target_class.device
        )
        distance_mask[:, self.distance_cols] = True
        distance_mask = distance_mask & target_class.bool()
        if distance_mask.any():
            distance_scale = distance_mask[:, self.distance_cols].float().mean()
            # scale the loss by the inverse ratio (as described above for angle loss)
            distance_loss = (
                prel_distance_loss(
                    input_param[distance_mask], target_param[distance_mask]
                )
                / distance_scale
            )
        else:
            distance_loss = torch.tensor(0.0)

        # combine all parametric loss functions
        return self.angle_weight * angle_loss + self.distance_weight * distance_loss
