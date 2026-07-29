# run inference on a dataset and return either an output file that is compatible with SGBench or Haystack (or raw data)
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import torch
import numpy as np
from sgadl import Config, from_config
from sgadl.data.iter_data import get_iter_loader
from sgadl.trainer import Trainer
from project_paths import project_paths


def convert_to_sgbench(data, graph_constraint):
    images = []
    for img_id, pairs, rel_outputs, instance_targets in zip(
        data["rel_img_ids"],
        data["pair_ids"],
        data["rel_outputs"],
        data["instance_targets"],
    ):
        # R@k, sort by no-rel, choose top predicate
        row_order = torch.argsort(rel_outputs[:, 0], descending=True)
        best_predicate = rel_outputs[:, 1:].argmax(dim=-1)
        triplets = torch.cat(
            (pairs[row_order], best_predicate[row_order][:, None]), dim=1
        )

        # compute instance -> category mapping
        # this is only implicitely stored in DSFormer's output
        inst2cat = {}
        for p, t in zip(
            torch.cat((pairs[:, 0], pairs[:, 1])),
            torch.cat((instance_targets[:, 0], instance_targets[:, 1])),
        ):
            # the instance classes are provided per relation
            # that means that they are duplicated
            # if everything is correct, there should be no mismatch between the instance targets
            if int(p) in inst2cat:
                assert inst2cat[int(p)] == int(t)
            else:
                inst2cat[int(p)] = int(t)

        num_inst = int(pairs.max()) + 1
        # DSFormer relies on segmentation masks, so no bounding box information is calculated here
        instances = [{"category": inst2cat[p]} for p in range(num_inst)]

        images.append(
            {
                "id": int(img_id),
                "seg_filename": f"{img_id}.tiff",
                "instances": instances,
                "triplets": triplets.tolist(),
            }
        )
    return {"version": 1, "images": images}


def convert_to_haystack(data: dict):
    predictions = torch.cat(data["rel_outputs"])
    sbjobj = torch.cat(data["pair_ids"]).to(dtype=predictions.dtype)
    img_ids = torch.repeat_interleave(
        torch.tensor(data["rel_img_ids"]),
        torch.tensor([len(p) for p in data["pair_ids"]]),
    ).to(dtype=predictions.dtype)
    return torch.cat((img_ids[:, None], sbjobj, predictions), dim=1).numpy()


class TestTrainer(Trainer):
    def _setup_loaders(
        self, config: Config, anno_path, img_dir, seg_dir, num_workers, debug_mode=False
    ):
        train_entries, node_names, rel_names = from_config.get_data_entries(
            config, anno_path=anno_path, split="train", debug_mode=debug_mode
        )
        val_entries, node_names, rel_names = from_config.get_data_entries(
            config,
            anno_path="/data/howto100m_features/lorenjul/datasets/haystack/v0_2.json",
            split="all",
            debug_mode=debug_mode,
        )
        self.train_loader = get_iter_loader(
            entries=train_entries,
            node_names=node_names,
            rel_names=rel_names,
            img_dir=img_dir,
            seg_dir=seg_dir,
            is_train=True,
            num_workers=num_workers,
            max_relations=self.rels_per_batch,
            augmentations=from_config.get_augmentations(config, split="train"),
            neg_ratio=config.neg_ratio,
            ignore_norel_predicates=config.ignore_norel_predicates,
            return_normals=config.data.normals,
            return_depth=config.data.depth,
            load_params=self._use_params,
        )

        self.val_loader = get_iter_loader(
            entries=val_entries,
            node_names=node_names,
            rel_names=rel_names,
            img_dir=img_dir,
            seg_dir=seg_dir,
            is_train=False,
            num_workers=num_workers,
            max_relations=self.rels_per_batch,
            augmentations=from_config.get_augmentations(config, split="val"),
            ignore_norel_predicates=config.ignore_norel_predicates,
            return_normals=config.data.normals,
            return_depth=config.data.depth,
            load_params=self._use_params,
        )


def cli():
    parser = ArgumentParser()
    parser.add_argument(
        "model", help="Model folder with config.json and best_state.pth"
    )
    parser.add_argument(
        "--sgbench", help="Output path for SGBench-compatible ZIP file."
    )
    parser.add_argument(
        "--haystack", help="Output path for Haystack-compatible NPY file."
    )
    parser.add_argument(
        "--raw", help="Output path to store raw outputs from evaluate()"
    )
    parser.add_argument("--anno")
    parser.add_argument("--img")
    parser.add_argument("--seg")
    parser.add_argument(
        "--ema",
        default=False,
        action="store_true",
        help="Whether to use the EMA model for evaluation.",
    )
    parser.add_argument(
        "--bs", default=None, type=int, help="Overwrite the rels_per_batch config"
    )
    parser.add_argument("--workers", default=10, type=int)
    args = parser.parse_args()

    assert args.sgbench or args.haystack or args.raw

    model_dir = Path(args.model)
    config = Config.from_file(model_dir / "config.json")
    checkpoint = torch.load(
        model_dir / "best_state.pth", map_location="cpu", weights_only=True
    )

    if args.ema and "model_ema" not in checkpoint:
        raise RuntimeError("No EMA model available in the checkpoint")

    # increase the batch size if desired
    if args.bs is not None:
        config.rels_per_batch = args.bs

    trainer = TestTrainer(
        anno_path=(
            project_paths.get_anno_path(config.data.source)
            if args.anno is None
            else args.anno
        ),
        img_dir=(
            project_paths.get_img_dir(config.data.source)
            if args.img is None
            else args.img
        ),
        seg_dir=(
            project_paths.get_seg_dir(config.data.source)
            if args.seg is None
            else args.seg
        ),
        config=config,
        out_dir=None,
        num_workers=args.workers,
        dump_data=False,
        start_state_dict=checkpoint,
    )

    _metrics, data = trainer.evaluate(epoch=0, use_ema=args.ema)

    # convert to triplets.json
    # create segmentation masks from ground truth
    # the idea is that not actual ground truth is used for this script but pseudo ground truth

    if args.raw:
        torch.save(data, args.raw)

    if args.sgbench:
        sgbench_triplets = convert_to_sgbench(data)
        with TemporaryDirectory(prefix="mlbl-") as tmp_dir:
            tmp_dir = Path(tmp_dir)
            # TODO: create tiff files from ground truth
            with open(tmp_dir / "triplets.json", "w") as f:
                # create triplets.json
                json.dump(sgbench_triplets, f)

            with ZipFile(args.sgbench, "w") as archive:
                for tiff in tmp_dir.glob("*.tiff"):
                    archive.write(tiff)
                archive.write(tmp_dir / "triplets.json")

    if args.haystack:
        # create NPY file for Haystack evaluation
        np.save(args.haystack, convert_to_haystack(data))

    print("done")


if __name__ == "__main__":
    cli()
