from argparse import ArgumentParser
import json
import torch
from tqdm import tqdm
from sgadl.config import Config
from sgadl.trainer import Trainer


def count_stats(trainer: Trainer):
    n_rel_val = 0
    counts_val = torch.zeros(
        len(trainer.val_loader.dataset.rel_names), dtype=torch.long
    )
    for x in tqdm(trainer.val_loader.dataset.base_dataset):
        counts_val += x["rels"].sum(0)
        n_rel_val += len(x["rels"])
    n_rel_train = 0
    counts_train = torch.zeros(
        len(trainer.train_loader.dataset.rel_names), dtype=torch.long
    )
    for x in tqdm(trainer.train_loader.dataset):
        counts_train += x["rels"].sum(0)
        n_rel_train += len(x["rels"])
    return {
        # loss weights
        "node_weight": trainer.node_criterion.weight.cpu().tolist(),
        "rel_pos_weight": trainer.rel_criterion.pos_weight.cpu().tolist(),
        # val set
        "n_rel_val": n_rel_val,
        "n_img_val": len(trainer.val_loader.dataset),
        "counts_val": counts_val.tolist(),
        # train set
        "n_rel_train": n_rel_train,
        "n_img_train": len(trainer.train_loader.dataset),
        "counts_train": counts_train.tolist(),
    }


def cli():
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "--anno", required=True, help="Path to PSG JSON annotation file"
    )
    parser.add_argument("--img", required=True, help="Directory that contains images")
    parser.add_argument(
        "--seg", required=True, help="Directory that contains segmentation masks"
    )
    parser.add_argument(
        "stats", help="Where to write the stats to. Will create a JSON file."
    )
    args = parser.parse_args()

    config = Config.from_file(args.config)

    trainer = Trainer(
        anno_path=args.anno,
        img_dir=args.img,
        seg_dir=args.seg,
        out_dir=None,
        config=config,
        num_workers=0,
    )

    stats = count_stats(trainer)

    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    cli()
