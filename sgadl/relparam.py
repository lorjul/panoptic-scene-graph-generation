from typing import Sequence
import torch

RELTYPE_IGNORE = 0
RELTYPE_ANGLE = 1
RELTYPE_DISTANCE = 2
RTYPE_DICT = {
    "NONE": RELTYPE_IGNORE,
    "i_behind": RELTYPE_ANGLE,
    "i_front": RELTYPE_ANGLE,
    "i_right": RELTYPE_ANGLE,
    "i_left": RELTYPE_ANGLE,
    "i_above": RELTYPE_ANGLE,
    "i_below": RELTYPE_ANGLE,
    "d_behind": RELTYPE_ANGLE,
    "d_front": RELTYPE_ANGLE,
    "d_right": RELTYPE_ANGLE,
    "d_left": RELTYPE_ANGLE,
    "i_distance": RELTYPE_DISTANCE,
    "i_touching": RELTYPE_IGNORE,
    "i_on": RELTYPE_IGNORE,
}

DIRECTIONAL_INSTANCES = (
    "chair",
    "bed",
    "fridge",
    "shelf",
    "chair",
    "dishwasher",
    "shelf",
    "shelf",
    "microwave",
    "mirror",
    "screen",
    "chair",
    "oven",
    "shelf",
    "shelf",
    "sink",
    "sofa",
    "sink",
    "screen",
    "shelf",
    "toilet",
    "art",
    "window",
)


def names_to_rel_types(names):
    return torch.tensor([RTYPE_DICT[n] for n in names], dtype=torch.long)


def relnames_to_cols(names):
    rel_types = names_to_rel_types(names)

    angle_cols = rel_types == RELTYPE_ANGLE
    distance_cols = rel_types == RELTYPE_DISTANCE

    return angle_cols, distance_cols


class TargetThresholder:
    def __init__(
        self,
        relnames: Sequence[str],
        min_ign_angle_deg: float,
        max_ign_angle_deg: float,
        min_ign_distance_m: float,
        max_ign_distance_m: float,
    ):
        self.first_norel = relnames[0] == "NONE"
        self.angle_cols, self.distance_cols = relnames_to_cols(relnames)
        self.min_ign_angle = torch.tensor(min_ign_angle_deg).deg2rad().cos()
        self.max_ign_angle = torch.tensor(max_ign_angle_deg).deg2rad().cos()
        self.min_ign_distance = min_ign_distance_m
        self.max_ign_distance = max_ign_distance_m

    def __call__(self, target_class: torch.Tensor, target_param: torch.Tensor):
        thresh_target = torch.full_like(target_class, fill_value=-1)

        # assign angle predicates
        sel = torch.zeros_like(target_class, dtype=torch.bool)
        sel[:, self.angle_cols] = (
            target_param[:, self.angle_cols] >= self.min_ign_angle
        ) & target_class[:, self.angle_cols].bool()
        thresh_target[sel] = 1
        # set to 0 if the parameter is set and above the max angle or if the parameter is not set
        sel[:, self.angle_cols] = (
            (target_param[:, self.angle_cols] <= self.max_ign_angle)
            & target_class[:, self.angle_cols].bool()
        ) | (~target_class[:, self.angle_cols].bool())
        thresh_target[sel] = 0

        # assign distance predicates
        sel = torch.zeros_like(target_class, dtype=torch.bool)
        sel[:, self.distance_cols] = (
            target_param[:, self.distance_cols] <= self.min_ign_distance
        ) & target_class[:, self.distance_cols].bool()
        thresh_target[sel] = 1
        # set to 0 if the parameter is set and above the max angle or if the parameter is not set
        sel[:, self.distance_cols] = (
            (target_param[:, self.distance_cols] >= self.max_ign_distance)
            & target_class[:, self.distance_cols].bool()
        ) | (~target_class[:, self.distance_cols].bool())
        thresh_target[sel] = 0

        # assign the non-distance/non-angle predicates
        sel = torch.ones_like(target_class, dtype=torch.bool)
        sel[:, self.distance_cols] = False
        sel[:, self.angle_cols] = False
        thresh_target[sel] = target_class[sel]

        # special handling for no-rel
        if self.first_norel:
            # update the no-relation class
            # From the dataset, it is 1 if was a sampled negative and 0 if any of the relation parameters are set.
            #
            # However, let's look at the following example:
            # All predicates are set to 0 except for one angle-predicate. But the angle is close to 90 degrees.
            # That means that it will be thresholded as a negative but the parameters will still be learned.
            # The no-relation class must reflect this situation correctly: The incoming target_class tensor will
            # have the no-relation class set to 0, because one of the predicates is set.
            # However, in the end, all predicates are set to 0, so the no-relation class must be set to 1 instead.
            #
            # Now, there is one corner case. What if all predicates are set to 0, except one predicate that is set to ignore.
            # Should we set the no-relation class to 0 or to 1? For R@k, selecting this relation makes no sense. Therefore,
            # we will set the no-relation class to 1.
            # Alternatively, we could argue that if some predicates are set to ignore, the no-relation class should also be
            # set to ignore because the predicates are not clearly defined.
            #
            # Ufft, that was something :)

            thresh_target[:, 0] = (thresh_target[:, 1:] != 1).all(1)
            # OR use the following if you want to set the no-relation class to ignore:
            # thresh_target[(thresh_target[:, 1:] != 1).all(1), 0] = -1
            # thresh_target[(thresh_target[:, 1:] == 0).all(1), 0] = 1
            # thresh_target[(thresh_target[:, 1:] == 1).any(1), 0] = 0

        return thresh_target
