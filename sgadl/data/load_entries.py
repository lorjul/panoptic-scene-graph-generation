from pathlib import Path
from typing import Literal, Optional
import json
import pickle
import numpy as np
import torch
from torchvision.ops import box_convert
from collections import defaultdict


def load_psg_entries(
    anno_path,
    split: Literal["train", "val", "test", "all"],
    max_entries: Optional[int] = None,
):
    """Loads entries, node_names, rel_names for the `SGDataset`.
    For semi-supervised learning, you probably want to create a new entry loader funciton.
    """
    entries = []

    with open(anno_path) as f:
        anno = json.load(f)

    if split == "all":
        test_ids = set()
        # val_ids = set()
    else:
        test_ids = set(anno["test_image_ids"])
        # val_ids = set(anno["test_image_ids"])
    for img in anno["data"]:
        img = dict(img)

        if len(img["annotations"]) == 0:
            continue

        # skip self-relations and increase rel index (0 is reserved for no-rel)
        if "relations" in img:
            if len(img["relations"]) == 0:
                # don't allow empty relations in trainin set
                continue

            param_relations = defaultdict(list)
            new_relations = []
            for sbj, obj, rel in img["relations"]:
                if sbj != obj:
                    new_relations.append((sbj, obj, rel + 1))
                    param_relations[(sbj, obj)].append(rel)
            if len(img["relations"]) > 0 and len(param_relations) == 0:
                # the only relation was a self-relation -> skip
                continue

            img.pop("relations")
            img["param_relations"] = [
                {"sbj": k[0], "obj": k[1], "rels": v}
                for k, v in param_relations.items()
            ]
        else:
            # no relation annotation is only allowed for all/test split
            assert split in ("all", "test"), split

        img_id = img["image_id"]
        in_test = img_id in test_ids
        # in_val = img_id in val_ids
        if split == "all":
            entries.append(img)
        elif split == "train" and not in_test:
            entries.append(img)
        elif split == "val" and in_test:
            entries.append(img)
        elif split == "test" and in_test:
            entries.append(img)

    assert len(entries) > 0, f"Empty dataset: {split}"

    node_names = anno["thing_classes"] + anno["stuff_classes"]
    rel_names = anno["predicate_classes"]
    if max_entries is not None:
        entries = entries[:max_entries]
    return entries, node_names, rel_names


def load_objects365_entries(anno_path, img_dir):
    """`img_dir` will be used to check if the image files exist"""
    img_dir = Path(img_dir)
    with open(anno_path) as f:
        raw_annos = json.load(f)

    assert all((i == x["id"] - 1 for i, x in enumerate(raw_annos["categories"])))
    node_names = [x["name"] for x in raw_annos["categories"]]
    rel_names = None

    anno_by_img = defaultdict(list)
    for a in raw_annos["annotations"]:
        anno_by_img[a["image_id"]].append(a)

    entries = []
    num_missing = 0
    for img in raw_annos["images"]:
        # check if file exists. TODO: download the missing files
        file_name = Path(img["file_name"]).name
        if not (img_dir / file_name).exists():
            num_missing += 1
            continue
        annos = []
        for a in anno_by_img[img["id"]]:
            x1, y1, w, h = a["bbox"]
            annos.append(
                {
                    "iscrowd": a["iscrowd"],
                    "bbox": (x1, y1, x1 + w, y1 + h),
                    "category_id": a["category_id"] - 1,
                }
            )
        entries.append(
            {
                "image_id": img["id"],
                "file_name": file_name,
                "annotations": annos,
            }
        )

    if num_missing > 0:
        print("Missing images:", num_missing)

    return entries, node_names, rel_names


def load_psgcoco_entries(anno_path, img_prefix, seg_prefix):
    """Loads all files from the coco folder of PSG. Even those that are not listed in the PSG annotation file"""
    with open(anno_path) as f:
        raw_annos = json.load(f)

    assert all((i == x["id"] - 1 for i, x in enumerate(raw_annos["categories"])))
    node_names = [x["name"] for x in raw_annos["categories"]]
    rel_names = None

    anno_by_img = {}
    for a in raw_annos["annotations"]:
        anno_by_img[a["image_id"]] = a

    entries = []
    for img in raw_annos["images"]:
        file_name = img["file_name"]
        pan_seg_file_name = anno_by_img[img["id"]]["file_name"]
        annos = []
        seg_info = []
        for a in anno_by_img[img["id"]]["segments_info"]:
            x1, y1, w, h = a["bbox"]
            annos.append(
                {
                    "iscrowd": a["iscrowd"],
                    "bbox": (x1, y1, x1 + w, y1 + h),
                    # starts at 1 in annotation file
                    "category_id": a["category_id"] - 1,
                }
            )
            seg_info.append({"id": a["id"]})
        entries.append(
            {
                "image_id": img["id"],
                "file_name": f"{img_prefix}/{file_name}",
                "pan_seg_file_name": f"{seg_prefix}/{pan_seg_file_name}",
                "annotations": annos,
                "segments_info": seg_info,
                # no relation information
            }
        )

    return entries, node_names, rel_names


def _vg_get_split(
    split_array: np.ndarray,
    split: Literal["train", "val", "test", "all"],
    variant: Literal["ietrans", "kaihua"],
):
    if split == "all":
        return np.ones(len(split_array), dtype=bool)

    # IETrans
    if variant == "ietrans":
        if split == "train":
            return split_array == 2
        if split == "val":
            return split_array == 1
        if split == "test":
            return split_array == 0

    # Kaihua-Tang
    if variant == "kaihua":
        if split == "train":
            return split_array == 0
        if split == "val":
            return split_array == 2

    raise RuntimeError(f"Unsupported VG split or variant: {split}@{variant}")


def load_visual_genome_entries(
    img_data_path,
    meta_path,
    h5_path,
    split: Literal["train", "val", "test", "all"],
    variant: Literal["ietrans", "kaihua"],
    max_entries: Optional[int] = None,
):
    """
    Loads relation entries from VG-150 data.

    :param img_data_path: Path to image_data.json
    :param meta_path: Path to ./50/VG-SGG-dicts-with-attri.json
    :param h5_path: Path to ./50/VG-SGG-with-attri.h5
    :param split: The data subset to choose from.
    :param max_entries: Whether to limit the number of returned entries. This is only really useful for debugging.
    """

    # h5py dependency is only required for visual genome
    try:
        import h5py
    except ImportError:
        raise ImportError("Loading Visual Genome requires h5py to be installed")

    # load metadata from JSON file
    with open(meta_path) as f:
        metadata = json.load(f)
    node_names = [
        metadata["idx_to_label"][str(i + 1)]
        for i in range(len(metadata["idx_to_label"]))
    ]
    rel_names = [
        metadata["idx_to_predicate"][str(i + 1)]
        for i in range(len(metadata["idx_to_predicate"]))
    ]

    with open(img_data_path) as f:
        image_data = json.load(f)
    image_ids = []
    image_sizes = []
    for d in image_data:
        image_ids.append(d["image_id"])
        image_sizes.append(max(d["width"], d["height"]))
    image_ids = np.array(image_ids)
    image_sizes = np.array(image_sizes)

    # load annotations
    entries = []
    with h5py.File(h5_path) as f:
        # use [:] to convert to numpy array
        group = _vg_get_split(f["split"][:], split=split, variant=variant)

        for img_id, img_size, first_box, last_box, first_rel, last_rel in zip(
            image_ids[group],
            image_sizes[group],
            f["img_to_first_box"][group],
            f["img_to_last_box"][group],
            f["img_to_first_rel"][group],
            f["img_to_last_rel"][group],
        ):
            if first_box == -1 or last_box == -1:
                continue
            boxes = []
            coords = f["boxes_1024"][first_box : last_box + 1] / 1024 * img_size
            # convert to xyxy format
            coords = box_convert(torch.tensor(coords), in_fmt="cxcywh", out_fmt="xyxy")
            # box labels start from 1 in the .h5 file
            categs = f["labels"][first_box : last_box + 1].squeeze(1) - 1
            for coord, categ in zip(coords, categs):
                boxes.append(
                    {
                        # convert to xyxy format
                        "bbox": coord.tolist(),
                        "category_id": int(categ),
                    }
                )

            pairs = f["relationships"][first_rel : last_rel + 1] - first_box
            # predicate labels start from 1 in the .h5 file
            predicates = f["predicates"][first_rel : last_rel + 1].squeeze(1) - 1
            param_relations = defaultdict(set)
            for pair, rel in zip(pairs, predicates):
                assert (pair < len(boxes)).all()
                param_relations[(int(pair[0]), int(pair[1]))].add(int(rel))

            if len(param_relations) == 0:
                # skip this entry, there are no relations
                continue

            entries.append(
                {
                    "image_id": int(img_id),
                    "file_name": f"{img_id}.jpg",
                    "annotations": boxes,
                    "param_relations": [
                        {"sbj": k[0], "obj": k[1], "rels": list(v)}
                        for k, v in param_relations.items()
                    ],
                    # no segmentation information available
                    "pan_seg_file_name": None,
                    "segments_info": None,
                }
            )

            if max_entries and len(entries) >= max_entries:
                break

    return entries, node_names, rel_names


def load_isg_entries(
    anno_path,
    split: Literal["train", "val", "test", "all"],
    debug_mode=False,
    ignore_nodes=None,
):
    print("Loading ISG entries...", end="", flush=True)
    if Path(anno_path).suffix == ".pkl":
        with open(anno_path, "rb") as f:
            anno = pickle.load(f)
    else:
        with open(anno_path) as f:
            anno = json.load(f)
    print("done", flush=True)

    if debug_mode:
        train_ids = set(anno["train_ids"][:10])
        val_ids = set(anno["val_ids"][:10])
        test_ids = set(anno.get("test_ids", [])[:10])
    else:
        train_ids = set(anno["train_ids"])
        val_ids = set(anno["val_ids"])
        test_ids = set(anno.get("test_ids", []))

    remap_cat = None
    if ignore_nodes is not None:
        for n in ignore_nodes:
            assert n in anno["node_names"]
        remap_cat = {}
        new_nodes = []
        for i, n in enumerate(anno["node_names"]):
            if n not in ignore_nodes:
                remap_cat[i] = len(remap_cat)
                new_nodes.append(n)
        anno["node_names"] = new_nodes

    # shift index by one
    entries = []
    for x in anno["data"]:
        if split == "train":
            if x["image_id"] not in train_ids:
                continue
        elif split == "val":
            if x["image_id"] not in val_ids:
                continue
        elif split == "test":
            if x["image_id"] not in test_ids:
                continue

        # remove the relations key, we don't want to use it accidentally
        if "relations" in x:
            x.pop("relations")
        if "ignore_rels" in x:
            x.pop("ignore_rels")

        if remap_cat:
            new_annotations = []
            new_segments_info = []
            remap_inst = {}
            for i, (a, s) in enumerate(zip(x["annotations"], x["segments_info"])):
                if a["category_id"] in remap_cat:
                    a["category_id"] = remap_cat[a["category_id"]]
                    new_annotations.append(a)
                    new_segments_info.append(s)
                    remap_inst[i] = len(remap_inst)
            x["annotations"] = new_annotations
            x["segments_info"] = new_segments_info
            new_rels = []
            for pr in x["param_relations"]:
                if pr["sbj"] in remap_inst and pr["obj"] in remap_inst:
                    pr["sbj"] = remap_inst[pr["sbj"]]
                    pr["obj"] = remap_inst[pr["obj"]]
                    new_rels.append(pr)
            x["param_relations"] = new_rels
        if len(x["param_relations"]) > 0:
            entries.append(x)

    return entries, list(anno["node_names"]), list(anno["rel_names"])


def load_psg_for_isg_entries(
    anno_path, depth_root: Path, predicate_mapping: dict, ignore_box_class=True
):
    """This is only intended as a test set"""
    """Loads entries, node_names, rel_names for the `SGDataset`.
    For semi-supervised learning, you probably want to create a new entry loader funciton.
    """
    entries = []

    with open(anno_path) as f:
        anno = json.load(f)

    pred_classes = anno["predicate_classes"]
    predicate_mapping = {pred_classes.index(k): v for k, v in predicate_mapping.items()}

    test_ids = set(anno["test_image_ids"])
    for img in anno["data"]:
        if img["image_id"] not in test_ids:
            continue

        img = dict(img)

        if len(img["annotations"]) == 0:
            continue

        if ignore_box_class:
            # map instance classes to the same id, we're not using them anyway
            new_annotations = []
            for ann in img["annotations"]:
                ann["category_id"] = 0
                new_annotations.append(ann)
            img["annotations"] = new_annotations

        # skip self-relations and increase rel index (0 is reserved for no-rel)
        if "relations" in img:
            param_rels = defaultdict(list)
            for sbj, obj, rel in img["relations"]:
                if sbj != obj and rel in predicate_mapping:
                    param_rels[(sbj, obj)].append(predicate_mapping[rel])
            if len(param_rels) == 0:
                # don't allow empty relations
                continue

            new_param_relations = []
            new_relations = []
            for (sbj, obj), v in param_rels.items():
                rels = []
                params = []
                for r in v:
                    rels.append(r)
                    params.append(0.0)
                    new_relations.append((sbj, obj, r))
                new_param_relations.append(
                    {"sbj": sbj, "obj": obj, "rels": rels, "params": params}
                )

            img["param_relations"] = new_param_relations
            img["relations"] = new_relations

        # assign depth if requested
        filename = img["file_name"].replace(".jpg", ".npy")
        img["depth_file_name"] = depth_root / filename

        entries.append(img)

    assert len(entries) > 0, "Empty dataset"

    return entries
