import random
from copy import deepcopy
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.multiprocessing as mp
from sgadl.data.data import SGDataset


class IdxStore:
    """This class provides the image IDs to the workers.
    It is very similar to a queue of IDs but I had some problems with queues which were blocking
    threads when the main process crashed. This class does not have this problem.
    """

    def __init__(self, num):
        self.manager = mp.Manager()
        self.num = num
        self.ids = self.manager.list(range(num))
        self.reset()
        self.lock = self.manager.Lock()

    def reset(self):
        new_ids = list(range(self.num))
        random.shuffle(new_ids)
        self.ids[:] = new_ids

    def get(self, timeout=None):
        with self.lock:
            if self.ids:
                return self.ids.pop()
            return None


class RelAccumDataset(IterableDataset):
    def __init__(self, base_dataset, max_relations: int):
        self.idx_queue = IdxStore(num=len(base_dataset))
        self.base_dataset = base_dataset
        self.max_relations = max_relations
        self._staged_batch = self._reset_staged()
        self._fetched_batch = None

    def _reset_staged(self):
        return {
            "idx": [],
            "image_id": [],
            "img": [],
            "segmentation": [],
            "num_instances": [],
            "bboxes": [],
            "box_categories": [],
            "total_rel_num": 0,
            "pairs": [],
            "rels": [],
            "rel_params": [],
        }

    def __iter__(self):
        return self

    def __next__(self):
        # while the staged batch is too small, keep increasing it
        while self._staged_batch["total_rel_num"] < self.max_relations:
            # fetch a new img batch if not present
            if self._fetched_batch is None:
                try:
                    # the queue is already full, no timeout should be required at all
                    idx = self.idx_queue.get(timeout=10)
                    # there are no more further samples
                    if idx is None:
                        break
                    self._fetched_batch = deepcopy(self.base_dataset[idx])
                except StopIteration:
                    # finalize the last batch; return what's left of the staged batch
                    # this is equivalent of a drop_last=False setting
                    # TODO: I think this try-except is unnecessary. Remove it if the NotImplementedError is not raised during training
                    raise NotImplementedError("Unexpected StopIteration")

            # determine how many relations can be appended from the fetched batch
            fetched_pairs = self._fetched_batch["pairs"]
            fetched_rels2 = self._fetched_batch["rels"]
            fetched_params = self._fetched_batch["rel_params"]
            num_fetched = len(fetched_pairs)
            missing = self.max_relations - self._staged_batch["total_rel_num"]
            num_append = min(num_fetched, missing)

            # append relations
            self._staged_batch["total_rel_num"] += num_append
            self._staged_batch["pairs"].append(fetched_pairs[:num_append])
            self._staged_batch["rels"].append(fetched_rels2[:num_append])
            self._staged_batch["rel_params"].append(fetched_params[:num_append])
            self._staged_batch["idx"].append(self._fetched_batch["idx"])
            self._staged_batch["image_id"].append(self._fetched_batch["image_id"])
            self._staged_batch["img"].append(self._fetched_batch["img"])
            # only add segmentation if the dataset provides it
            if self._fetched_batch.get("segmentation") is not None:
                self._staged_batch["segmentation"].append(
                    self._fetched_batch["segmentation"]
                )
            self._staged_batch["bboxes"].append(self._fetched_batch["bboxes"])
            self._staged_batch["box_categories"].append(
                self._fetched_batch["box_categories"]
            )

            if num_append == num_fetched:
                # fetched batch is empty
                self._fetched_batch = None
            else:
                # remove the first entries
                self._fetched_batch["pairs"] = fetched_pairs[num_append:]
                self._fetched_batch["rels"] = fetched_rels2[num_append:]
                self._fetched_batch["rel_params"] = fetched_params[num_append:]

        # no last data to return and no new data to come => the end is here!
        if self._staged_batch["total_rel_num"] == 0:
            raise StopIteration()

        out_batch = {
            "idx": torch.tensor(self._staged_batch["idx"]),
            "image_id": torch.tensor(self._staged_batch["image_id"]),
            "img": torch.stack(self._staged_batch["img"]),
            "num_instances": torch.tensor(
                [len(b) for b in self._staged_batch["bboxes"]]
            ),
            "bboxes": torch.cat(self._staged_batch["bboxes"]),
            "box_categories": torch.cat(self._staged_batch["box_categories"]),
            "num_relations": torch.tensor(
                [len(r) for r in self._staged_batch["pairs"]]
            ),
            "pairs": torch.cat(self._staged_batch["pairs"]),
            "rels": torch.cat(self._staged_batch["rels"]),
            "rel_params": torch.cat(self._staged_batch["rel_params"]),
        }

        # only add segmentation if the dataset provides it
        if self._staged_batch["segmentation"]:
            out_batch["segmentation"] = torch.cat(self._staged_batch["segmentation"])

        # staged batch is complete and will be sent as out_batch
        # now, start with the next batch, so a new fresh staged batch
        # note that self._fetched_batch will still contain remaining data for the next staged batch
        self._staged_batch = self._reset_staged()
        return out_batch

    @property
    def node_names(self):
        return self.base_dataset.node_names

    @property
    def rel_names(self):
        return self.base_dataset.rel_names

    def get_predicate_neg_ratio(self, *args, **kwargs):
        return self.base_dataset.get_predicate_neg_ratio(*args, **kwargs)

    def count_nodes(self):
        return self.base_dataset.count_nodes()


def _customdataloader_collate_fn(samples):
    """Had to use an extra function because it complained about not being able to pickle a lambda function."""
    return samples[0]


class CustomDataLoader(DataLoader):
    def __init__(
        self, *, dataset: RelAccumDataset, shuffle=True, num_workers=0, pin_memory=False
    ):
        assert isinstance(dataset, RelAccumDataset)
        assert shuffle, "CustomDataLoader only supports shuffled iteration"
        super().__init__(
            dataset=dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=_customdataloader_collate_fn,
            pin_memory=pin_memory,
        )

    def __iter__(self):
        # this dataloader only supports RelAccumDataset
        assert isinstance(self.dataset, RelAccumDataset)
        # setup queue
        self.dataset.idx_queue.reset()
        return super().__iter__()


def get_iter_loader(
    entries,
    node_names,
    rel_names,
    img_dir,
    seg_dir,
    is_train,
    augmentations,
    num_workers,
    max_relations: int,
    neg_ratio=None,
    ignore_norel_predicates=False,
    return_normals=False,
    return_depth=False,
    depth_dir=None,
    load_params=True,
):
    dataset = SGDataset(
        is_train=is_train,
        entries=entries,
        node_names=node_names,
        rel_names=rel_names,
        img_dir=img_dir,
        seg_dir=seg_dir,
        depth_dir=depth_dir,
        augmentations=augmentations,
        neg_ratio=neg_ratio,
        ignore_norel_predicates=ignore_norel_predicates,
        return_normals=return_normals,
        return_depth=return_depth,
        load_params=load_params,
    )
    iter_dataset = RelAccumDataset(base_dataset=dataset, max_relations=max_relations)

    return CustomDataLoader(
        dataset=iter_dataset, num_workers=num_workers, pin_memory=True
    )
