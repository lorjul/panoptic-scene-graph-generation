from typing import Iterable, Sequence
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import numpy as np
import torch
from torchvision.io import decode_image, ImageReadMode
from torchvision import tv_tensors
from torchvision.transforms.v2 import Compose

from sgadl.utils import open_segmask
from sgadl.data.rel_sample import sample_pairs
from sgadl.relparam import relnames_to_cols
from sgadl.data.depth_to_normal import depth_to_surface_normal


def make_multilabel_target(raw_rels, num_classes: int, return_params: bool):
    assert len(raw_rels) > 0
    pairs = []
    rels = []
    params = []
    for x in raw_rels:
        pairs.append((x["sbj"], x["obj"]))
        mask = torch.tensor(x["rels"])
        row = torch.zeros(num_classes, dtype=torch.long)
        row[mask] = 1
        rels.append(row)
        if return_params:
            p = torch.zeros(row.shape, dtype=torch.float)
            p[mask] = torch.tensor(x["params"])
            params.append(p)

    if return_params:
        params = torch.stack(params)
    else:
        # replace with dummy values
        params = torch.zeros_like(torch.stack(rels), dtype=torch.float)

    return torch.tensor(pairs), torch.stack(rels), params


def count_nodes(entries, num_node_outputs: int):
    counts = torch.zeros(num_node_outputs, dtype=torch.long)
    for entry in entries:
        for box in entry["annotations"]:
            counts[box["category_id"]] += 1
    return counts


def get_predicate_neg_ratio(entries, num_rel_classes: int, neg_ratio: float):
    pos = torch.zeros(num_rel_classes, dtype=torch.long)
    neg = torch.zeros(num_rel_classes, dtype=torch.long)
    num_relations = 0
    for entry in entries:
        for r in entry["param_relations"]:
            num_relations += 1

            this_rel = torch.zeros(num_rel_classes, dtype=torch.bool)
            this_rel[torch.tensor(r["rels"])] = True

            pos += this_rel
            neg += ~this_rel

    ratio = pos.float() / (neg.float() + round(neg_ratio * num_relations))

    # explicit none is more of a hack currently
    return torch.cat((torch.tensor((neg_ratio,)), ratio))


def get_predicate_neg_ratio_legacy(entries, num_rel_classes: int, neg_ratio):
    """This is the (broken) old implementation that somehow still works.
    TODO: investigate why that's the case.
    """
    pos = torch.zeros(num_rel_classes, dtype=torch.long)
    neg = torch.zeros(num_rel_classes, dtype=torch.long)
    for entry in entries:
        this_rel = torch.zeros(num_rel_classes, dtype=torch.bool)
        for r in entry["param_relations"]:
            for rel in r["rels"]:
                this_rel[rel] = True

        pos += this_rel
        neg += ~this_rel

    ratio = pos.float() / neg.float()

    assert neg_ratio is not None
    # explicit none is more of a hack currently
    return torch.cat((torch.tensor((neg_ratio,)), ratio))


def _convert_angles(angles: torch.Tensor):
    assert (angles <= 90).all()
    assert (angles >= 0).all()
    rad = torch.deg2rad(angles)
    return torch.clamp(torch.cos(rad), 0, 1)


class SGConcatDataset(ConcatDataset):
    """Subclass of `ConcatDataset` that also contains the `node_names` and `rel_names` properties.
    This dataset class is intended for semi-supervised learning.
    If you want to access the `count_nodes()` or `get_predicate_neg_ratio()` functions,
    call them directly on one of the child datasets.
    """

    def __init__(self, datasets: Iterable["SGDataset"]):
        super().__init__(datasets)
        nn = self.datasets[0].node_names
        rn = self.datasets[0].rel_names
        for d in self.datasets:
            assert d.node_names == nn
            assert d.rel_names == rn

    @property
    def node_names(self):
        return self.datasets[0].node_names

    @property
    def rel_names(self):
        return self.datasets[0].rel_names


class SGDataset(Dataset):
    def __init__(
        self,
        is_train: bool,
        entries: Sequence[dict],
        node_names: Sequence[str],
        rel_names: Sequence[str],
        img_dir,
        seg_dir,
        augmentations,
        neg_ratio=None,
        ignore_norel_predicates=False,
        return_depth=False,
        return_normals=False,
        depth_dir=None,
        load_params=True,
    ):
        """Creates a new SGDataset object
        :param is_train: Whether the dataset is for training data or not.
            In training mode, different augmentation and sampling is used.
        :param entries: Sequence of dictionaries that follow the PSG file format (data key).
            Per entry dict, the following keys are required: file_name, annotations, image_id.
        :param node_names: Class names of the node categories.
            For PSG, you get them from `thing_classes` and `stuff_classes`.
        :param rel_names: Sequence of predicate class names. This must not include a no-relation
            class name (will be added in the dataset).
        :param img_dir: Path to image directory such that `{img_dir}/{entry["file_name"]}` works.
        :param seg_dir: Path to segmentation masks directory such that `{seg_dir}/{entry["pan_seg_file_name"]}` works.
        :param neg_ratio: Negative ratio for random sampling. Only applies when `is_train == True`.
        """
        self.img_dir = Path(img_dir)
        self.seg_dir = None if seg_dir is None else Path(seg_dir)
        self.depth_dir = self.seg_dir if depth_dir is None else Path(depth_dir)
        self.node_names = node_names
        self.rel_names = ["NONE"] + rel_names
        self.entries = entries

        assert len(self.entries) > 0, "Empty dataset"

        if return_normals:
            raise NotImplementedError("Surface normals are currently not supported")
        self.tfm = Compose(augmentations)

        if is_train:
            assert isinstance(neg_ratio, (int, float)), neg_ratio
            self.neg_ratio = neg_ratio
        else:
            self.neg_ratio = None

        self.ignore_norel_predicates = ignore_norel_predicates

        self.load_depth = return_depth
        self.load_normals = return_normals
        self.load_params = load_params

        if self.load_params:
            # assign relation types
            self._is_angle, self._is_distance = relnames_to_cols(self.rel_names[1:])
        else:
            self._is_angle = torch.zeros((len(self.rel_names[1:]),), dtype=torch.long)
            self._is_distance = torch.zeros_like(self._is_angle)

        # change to npz if desired
        self.depth_format = "npy"

    def __len__(self):
        return len(self.entries)

    def _open_img2(self, file_name):
        return decode_image(str(self.img_dir / file_name), ImageReadMode.RGB)

    def _load_depth(self, file_name):
        if self.depth_format == "npz":
            depth = np.load((self.img_dir / file_name).with_suffix(".npz"))["depth"]
        else:
            depth = np.load(self.img_dir / file_name)
        # convert to tensor for compatibility with torchvision transforms
        return torch.tensor(depth)

    def _load_seg(self, pan_seg_file_name, segments_info):
        seg_classes = torch.from_numpy(open_segmask(self.seg_dir / pan_seg_file_name))
        seg_ids = torch.tensor([int(info["id"]) for info in segments_info])
        seg_masks = seg_classes == seg_ids[:, None, None]
        return seg_masks

    def __getitem__(self, idx):
        entry = self.entries[idx]
        categories = torch.tensor([b["category_id"] for b in entry["annotations"]])
        has_seg = entry.get("pan_seg_file_name") is not None
        if has_seg:
            seg_masks = self._load_seg(
                entry["pan_seg_file_name"], entry["segments_info"]
            )
        else:
            seg_masks = None

        img = self._open_img2(entry["file_name"])
        bboxes = tv_tensors.BoundingBoxes(
            torch.tensor([b["bbox"] for b in entry["annotations"]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=img.shape[-2:],
        )
        d = {"rgb": img, "box": bboxes}
        if has_seg:
            d["seg"] = tv_tensors.Mask(seg_masks)

        if self.load_depth or self.load_normals:
            # surface normals will be derived from depth
            # I couldn't get the blender surface normals to look like the dervied ones
            # since we want to generalise to derived normals for real-world data, deriving
            # the normals for synthetic data seems a good option
            if "depth_file_name" in entry:
                depth_file_name = entry["depth_file_name"]
            else:
                depth_file_name = (
                    entry["file_name"].replace("Image", "Depth").replace(".jpg", ".npy")
                )
            depth = self._load_depth(depth_file_name)

            if self.load_depth:
                d["depth"] = tv_tensors.Mask(depth)

            if self.load_normals:
                normals = torch.from_numpy(
                    depth_to_surface_normal(depth.numpy(), fx=600, fy=600)
                )
                # make normals channel-first
                normals = normals.permute(2, 0, 1)
                d["normals"] = tv_tensors.Image(normals)

        out = self.tfm(d)
        seg_masks = out.get("seg")
        bboxes = out["box"]

        # TODO: in the future, the model should decide how to load the depth map
        # converting it to RGB is just a simple hack in the meantime
        if self.load_depth:
            img = torch.stack((out["depth"], out["depth"], out["depth"]), dim=0)
        else:
            img = out["rgb"]

        raw_relations = entry["param_relations"]

        num_outputs = len(self.rel_names) - 1
        pos_pairs, rel_targets, rel_params = make_multilabel_target(
            raw_relations, num_classes=num_outputs, return_params=self.load_params
        )
        # convert angles to [0,1] using cos
        rel_params[:, self._is_angle] = _convert_angles(rel_params[:, self._is_angle])
        # we could also normalize distances here but I don't think that's necessary

        sampled_pairs = sample_pairs(
            num_instances=len(bboxes), pos_pairs=pos_pairs, neg_ratio=self.neg_ratio
        )

        sampled_targets = torch.zeros(
            (len(sampled_pairs), num_outputs + 1), dtype=torch.long
        )
        sampled_params = torch.zeros(sampled_targets.shape, dtype=torch.float)
        # set positive ground truth (comes first; see sample_pairs() for more information)
        # skip the no-relation label (defaults to 0)
        sampled_targets[: len(pos_pairs), 1:] = rel_targets
        sampled_params[: len(pos_pairs), 1:] = rel_params
        # set negative ground truth (comes second)
        # set the no-relation label
        sampled_targets[len(pos_pairs) :, 0] = 1
        if self.ignore_norel_predicates:
            sampled_targets[len(pos_pairs) :, 1:] = -1

        if self.neg_ratio is None:
            # if neg_ratio is None, it's not trainig
            # sort the pairs
            assert sampled_pairs.shape[1] == 2
            ordering = (
                sampled_pairs[:, 0] * (sampled_pairs.max() + 1) + sampled_pairs[:, 1]
            ).argsort()
        else:
            # shuffle relations
            # they will be split by the RelAccumDataset, so we want to avoid having all positive pairs
            # in one batch and all negatives in the other
            ordering = torch.randperm(len(sampled_targets))
        sampled_pairs = sampled_pairs[ordering]
        sampled_targets = sampled_targets[ordering]
        sampled_params = sampled_params[ordering]

        output = {
            "idx": torch.tensor(idx),
            "image_id": torch.tensor(int(entry["image_id"])),
            "img": img,
            "bboxes": bboxes,
            "segmentation": seg_masks,
            "box_categories": categories,
            "pairs": sampled_pairs,
            "rels": sampled_targets,
            "rel_params": sampled_params,
        }

        return output

    def __add__(self, other):
        if isinstance(other, SGDataset):
            return SGConcatDataset([self, other])
        return super().__add__(other)

    def count_nodes(self):
        return count_nodes(self.entries, len(self.node_names))

    def get_predicate_neg_ratio(self, legacy=False):
        if legacy:
            return get_predicate_neg_ratio_legacy(
                self.entries,
                num_rel_classes=len(self.rel_names) - 1,
                neg_ratio=self.neg_ratio,
            )
        return get_predicate_neg_ratio(
            self.entries,
            num_rel_classes=len(self.rel_names) - 1,
            neg_ratio=self.neg_ratio,
        )
