# use relation models but completely ignore the relation output and only train the subject and object classification
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from project_paths import project_paths
from ..config import Config
from .. import from_config
from ..trainer import Trainer, prepare_batch, confusion_matrix
from ..data.pretrain_pair_data import PretrainPairDataset
from ..data.common import get_generic_loader


class PretrainPairsTrainer(Trainer):
    """Subclass of `Trainer` that is intended to work with pair classification only. No relations will be trained here."""

    def __init__(
        self,
        anno_path,
        img_dir,
        seg_dir,
        config: Config,
        pairs_per_img: int,
        out_dir=None,
        num_workers=0,
        dump_data=True,
        start_state_dict=None,
        hide_batch_progress=False,
    ):
        self.pairs_per_img = pairs_per_img
        super().__init__(
            anno_path,
            img_dir,
            seg_dir,
            config,
            out_dir,
            num_workers,
            dump_data,
            start_state_dict,
            hide_batch_progress,
        )
        # relation metrics don't make sense here; look at nodes instead
        self.critical_metric = "node_acc/mean"

    def _setup_loaders(self, config: Config, anno_path, img_dir, seg_dir, num_workers):
        # I know, we are loading the entries twice, but this way it's easier to implement without changing too much stuff
        train_entries, node_names, rel_names = from_config.get_data_entries(
            config, anno_path=anno_path, split="train"
        )
        val_entries, node_names, rel_names = from_config.get_data_entries(
            config, anno_path=anno_path, split="val"
        )

        # overwrite data loaders with pair-only data
        self.train_loader = get_generic_loader(
            dataset=PretrainPairDataset(
                entries=train_entries,
                node_names=node_names,
                rel_names=rel_names,
                img_dir=img_dir,
                seg_dir=seg_dir,
                augmentations=from_config.get_augmentations(config, split="train"),
                pairs_per_img=self.pairs_per_img,
                rnd_len=40000,
            ),
            batch_size=config.batch_size,
            num_workers=num_workers,
            is_train=True,
        )
        self.val_loader = get_generic_loader(
            dataset=PretrainPairDataset(
                entries=val_entries,
                node_names=node_names,
                rel_names=rel_names,
                img_dir=img_dir,
                seg_dir=seg_dir,
                augmentations=from_config.get_augmentations(config, split="val"),
                pairs_per_img=self.pairs_per_img,
            ),
            batch_size=config.batch_size,
            num_workers=num_workers,
            is_train=False,
        )

    @torch.inference_mode()
    def evaluate(self, epoch: int):
        self.model.eval()

        node_losses = []

        # torch.cat is required at the end
        # it can also be the case that some images appear twice in the list
        all_node_targets = []
        all_node_outputs = []

        batch_iterator = split_batch_iter(
            tqdm(
                self.val_loader,
                leave=False,
                desc="eval",
                dynamic_ncols=True,
                disable=self.hide_batch_progress,
            ),
            max_relations=self.rels_per_batch,
        )
        for batch in batch_iterator:
            model_input, sbj_target, obj_target, rel_target = prepare_batch(
                batch, self.device
            )

            sbj_out, obj_out, _ = self.model(model_input)

            # node loss
            sbj_loss = self.node_criterion(sbj_out, sbj_target)
            obj_loss = self.node_criterion(obj_out, obj_target)
            node_loss = sbj_loss + obj_loss
            all_node_targets.append(sbj_target.cpu().clone())
            all_node_targets.append(obj_target.cpu().clone())
            all_node_outputs.append(sbj_out.argmax(dim=-1).cpu().clone())
            all_node_outputs.append(obj_out.argmax(dim=-1).cpu().clone())

            if torch.isnan(node_loss):
                if self.out_dir is not None:
                    torch.save(
                        {
                            "sbj_out": sbj_out,
                            "obj_out": obj_out,
                            "sbj_tgt": sbj_target,
                            "obj_tgt": obj_target,
                        },
                        self.out_dir / "dbg-nanloss.pth",
                    )
                raise RuntimeError()

            node_losses.append(node_loss.cpu().clone())

        # calculate per class accuracy
        all_node_targets = torch.cat(all_node_targets)
        all_node_outputs = torch.cat(all_node_outputs)

        per_class_node_acc = confusion_matrix(
            all_node_targets,
            all_node_outputs,
            normalize="true",
        ).diagonal()

        metrics = {
            "epoch_loss/val/node": torch.tensor(node_losses).mean(),
            "node_acc/mean": per_class_node_acc.mean(),
        }

        for name, acc in zip(self.val_loader.dataset.node_names, per_class_node_acc):
            metrics[f"node_class_acc/{name}"] = acc

        data = {"node_targets": all_node_targets, "node_outputs": all_node_outputs}

        return metrics, data


def cli():
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--pairs", default=1000, type=int)
    parser.add_argument("--anno", default=None)
    parser.add_argument("--img", default=None)
    parser.add_argument("--seg", default=None)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument("--no-batch-pbar", default=False, action="store_true")
    args = parser.parse_args()

    if args.anno is None:
        anno_path = project_paths.psg_annotation_dir
        print("Anno Path:", anno_path)
    else:
        anno_path = args.anno
    if args.img is None:
        img_dir = project_paths.psg_img_dir
        print("Img Dir:", img_dir)
    else:
        img_dir = args.img
    if args.seg is None:
        seg_dir = project_paths.psg_seg_dir
        print("Seg Dir:", seg_dir)
    else:
        seg_dir = args.seg

    config = Config.from_file(args.config)
    # only train with node information
    config.rel_weight = 0.0
    config.node_loss_weight = 1.0

    trainer = PretrainPairsTrainer(
        anno_path=anno_path,
        img_dir=img_dir,
        seg_dir=seg_dir,
        out_dir=args.output,
        config=config,
        pairs_per_img=args.pairs,
        num_workers=args.workers,
        hide_batch_progress=args.no_batch_pbar,
    )
    trainer.run(epochs=args.epochs)

    print(
        "Best value for ",
        trainer.critical_metric,
        ": ",
        trainer.best_metric_value,
        sep="",
    )


if __name__ == "__main__":
    cli()
