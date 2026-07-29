# dataset for pair pretraining
import random
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from torch.utils.data import Dataset
from .data import count_nodes
from ..utils import open_segmask
from .rel_sample import idx_to_sbjobj


class PretrainPairDataset(Dataset):
    def __init__(
        self,
        entries,
        node_names,
        rel_names,
        img_dir,
        seg_dir,
        augmentations,
        pairs_per_img: int,
        rnd_len: Optional[int] = None,
    ):
        """
        :param rnd_len: Randomly sample some images from the dataset. Useful when the dataset is very large (e.g. Objects365) or for testing.
        """
        self.entries = entries
        self.node_names = node_names
        self.rel_names = ["NONE"] + rel_names
        self.img_dir = Path(img_dir)
        self.seg_dir = Path(seg_dir)
        self.tfm = augmentations
        self.pairs_per_img = pairs_per_img
        self.rnd_len = rnd_len
        if self.rnd_len is not None:
            assert isinstance(self.rnd_len, int)
            self.rnd_len = min(self.rnd_len, len(self.entries))
            assert self.rnd_len <= len(self.entries)
            print("Rnd len is:", self.rnd_len)
            print("Max would be:", len(self.entries))

    def __len__(self):
        if self.rnd_len is None:
            return len(self.entries)
        return self.rnd_len

    def __getitem__(self, idx):
        raise NotImplementedError("TODO: modify such that we can use params")
        if self.rnd_len is not None:
            idx = random.randint(0, self.rnd_len - 1)

        entry = self.entries[idx]
        img = Image.open(self.img_dir / entry["file_name"]).convert("RGB")
        bboxes = torch.tensor([b["bbox"] for b in entry["annotations"]])

        seg_classes = torch.from_numpy(
            open_segmask(self.img_dir / entry["pan_seg_file_name"])
        )
        seg_ids = torch.tensor([info["id"] for info in entry["segments_info"]])
        seg = seg_classes == seg_ids[:, None, None]

        for tfm in self.tfm:
            img, seg, bboxes = tfm(img, seg, bboxes)

        num_nodes = len(bboxes)
        num_possibilities = num_nodes * (num_nodes - 1)
        idx = torch.randperm(num_possibilities)[: self.pairs_per_img]
        pairs = idx_to_sbjobj(idx, num_nodes)
        pseudo_relations = torch.cat(
            (pairs, torch.ones((len(pairs), len(self.rel_names)), dtype=pairs.dtype)),
            dim=1,
        )

        return {
            "image_id": int(entry["image_id"]),
            "img": img,
            "bboxes": bboxes,
            "box_categories": torch.tensor(
                [b["category_id"] for b in entry["annotations"]]
            ),
            "segmentation": seg,
            "sampled_relations": pseudo_relations,
        }

    def count_nodes(self):
        return count_nodes(self.entries, len(self.node_names))

    def get_predicate_neg_ratio(self):
        # this method is not really required but we need some dummy values for API compatibility
        return torch.ones(len(self.rel_names))
