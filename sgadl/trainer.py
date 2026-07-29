from pathlib import Path
import json
from typing import Tuple
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .data.iter_data import get_iter_loader
from .metrics import build_rel_metrics_dict
from .isg_metrics import build_isg_metrics_dict
from .config import Config
from . import from_config
from .utils import get_device, get_git_changes, get_git_commit
from .loss import (
    get_node_criterion,
    # get_multi_rel_criterion,
    get_thresh_criterion,
    BCEWithLogitsIgnoreNegative,
    ParamRelLoss,
)


class NoTensorboard:
    def add_scalar(self, *args, **kwargs):
        pass


def prepare_batch(
    batch: dict, device: torch.device
) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Takes a batch from data loader and returns the input to a model and the expected targets
    :param batch: The batch that comes directly from the data loader.
    :return: A 5-tuple of (model_input, sbj_target, obj_target, rel_target, param_target)
    """

    # TODO: make .to(device) non-blocking?
    model_input = {"img": batch["img"].to(device)}
    if "segmentation" in batch:
        model_input["segmentation"] = batch["segmentation"].to(device)
    box_target = batch["box_categories"].to(device)

    pair_ids_no_offset = batch["pairs"]
    # pair ids must be shifted by the number of instances
    shifted_num_inst = torch.cat((torch.tensor((0,)), batch["num_instances"][:-1]))
    pair_offset = shifted_num_inst.cumsum(dim=0)
    pair_ids = (
        pair_ids_no_offset
        + torch.repeat_interleave(pair_offset, batch["num_relations"])[:, None]
    )

    model_input["bboxes"] = batch["bboxes"]
    model_input["inst2img"] = torch.repeat_interleave(batch["num_instances"])
    model_input["pair_ids"] = pair_ids
    model_input["box_categories"] = box_target

    sbj_target = box_target[pair_ids[:, 0]]
    obj_target = box_target[pair_ids[:, 1]]

    rel_target = batch["rels"]
    param_target = batch["rel_params"]

    return model_input, sbj_target, obj_target, rel_target.float(), param_target


class Trainer:
    def __init__(
        self,
        anno_path,
        img_dir,
        seg_dir,
        config: Config,
        out_dir=None,
        num_workers=0,
        dump_data=True,
        start_state_dict=None,
        hide_batch_progress=False,
        debug_mode=False,
        is_transfer_learning=False,
    ):
        if debug_mode:
            print(
                "WARNING: debug mode is active. This will train with a much reduced training set. Some metrics will fail."
            )
        self._debug_mode = debug_mode
        self.device = get_device()
        self.rel_loss_weight = config.loss.rel_weight
        self.node_loss_weight = config.loss.node_loss_weight

        self.dump_data = dump_data
        self.grad_accumulate = config.grad_accumulate
        assert isinstance(self.grad_accumulate, int) and self.grad_accumulate > 0

        self.rels_per_batch = config.rels_per_batch

        self._use_params = config.data.source == "isg"
        self._setup_loaders(
            config, anno_path, img_dir, seg_dir, num_workers, debug_mode=debug_mode
        )

        self.model = from_config.get_model(
            config,
            num_node_outputs=len(self.train_loader.dataset.node_names),
            # rel_names includes the NO-REL class
            num_rel_outputs=len(self.train_loader.dataset.rel_names),
        )
        if start_state_dict is not None:
            with torch.no_grad():
                if is_transfer_learning:
                    print("Transfer learning initialisation")
                    # allow mismatching weight shapes for the final layers
                    self.model.load_state_dict_partial(start_state_dict["model"])
                else:
                    print("Resume training from checkpoint")
                    self.model.load_state_dict(start_state_dict["model"])
        self.model.to(self.device)
        self.ema_model = from_config.get_ema_model(config, self.model, self.device)
        if (
            self.ema_model is not None
            and start_state_dict is not None
            and "ema_model" in start_state_dict
        ):
            # TODO: is this correct?
            with torch.no_grad():
                self.ema_model.load_state_dict(start_state_dict["ema_model"])

        self.node_criterion = get_node_criterion(
            self.train_loader.dataset.count_nodes()
        ).to(self.device)

        # TargetThresholder is used to create binary labels based on the relation parameters
        # this is used for the relation loss and the metrics
        if self._use_params:
            self.rel_thresher = from_config.get_rel_thresher(
                config, relnames=self.train_loader.dataset.rel_names
            )
        else:
            self.rel_thresher = None

        if config.ignore_norel_predicates:
            raise NotImplementedError()
            self.rel_criterion = BCEWithLogitsIgnoreNegative(
                pos_weight=1
                / self.train_loader.dataset.get_predicate_neg_ratio(
                    legacy=config.loss.legacy_balancing
                )
            ).to(self.device)
        else:
            pos_neg_ratios = self.train_loader.dataset.get_predicate_neg_ratio(
                legacy=config.loss.legacy_balancing
            )
            if debug_mode:
                pos_neg_ratios = torch.ones_like(pos_neg_ratios)
            self.rel_criterion = get_thresh_criterion(
                rel_thresher=self.rel_thresher,
                pos_neg_ratios=pos_neg_ratios,
            ).to(self.device)

        if self._use_params:
            # creates parameter-specific loss functions for each relation
            self.param_criterion = ParamRelLoss(
                rel_names=self.train_loader.dataset.rel_names,
                angle_weight=config.loss.angle_weight,
                distance_weight=config.loss.distance_weight,
            )
        else:
            self.param_criterion = None

        if config.lr_backbone is None:
            self.optimizer = optim.AdamW(
                params=self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            extractor_params = []
            other_params = []
            for n, p in self.model.named_parameters():
                if "extractor" in n:
                    extractor_params.append(p)
                else:
                    other_params.append(p)

            optim_params = [{"params": other_params}]
            if config.lr_backbone > 0:
                optim_params.append(
                    {"params": extractor_params, "lr": config.lr_backbone}
                )
            self.optimizer = optim.AdamW(
                params=optim_params,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

        self.lr_scheduler = from_config.get_lr_scheduler(config, self.optimizer)

        self.grad_clipping = config.grad_clipping
        assert (
            self.grad_clipping is None or self.grad_clipping > 0
        ), "Gradient clipping must be None or a positive float value."

        self.best_metric_value = None
        if self._use_params:
            self.critical_metric = "isg_mAP"
        else:
            self.critical_metric = "rel_mean_recall/50"

        # for tensorboard
        self._global_step = 0

        # don't initialise state from checkpoint if set to transfer learning
        if start_state_dict is not None and not is_transfer_learning:
            self.optimizer.load_state_dict(start_state_dict["optim"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(start_state_dict["lr_scheduler"])
            self._global_step = start_state_dict["_global_step"]

        if out_dir is None:
            self.out_dir = None
            self.tensorboard = NoTensorboard()
        else:
            self.out_dir = Path(out_dir)
            self.tensorboard = SummaryWriter(log_dir=out_dir)
            config.to_file(self.out_dir / "config.json")
            self.tensorboard.add_text("config", config.to_markdown())

            git_commit = get_git_commit()
            if git_commit:
                with open(self.out_dir / "commit.txt", "w") as f:
                    f.write(git_commit)
                self.tensorboard.add_text("commit", git_commit)
                changes = get_git_changes()
                if changes:
                    with open(self.out_dir / "changes.patch", "w") as f:
                        f.write(changes)

        self.hide_batch_progress = hide_batch_progress

        self._hack_vg_nan_ok = config.data.source == "vg-ietrans"

    def _setup_loaders(
        self, config: Config, anno_path, img_dir, seg_dir, num_workers, debug_mode=False
    ):
        train_entries, node_names, rel_names = from_config.get_data_entries(
            config, anno_path=anno_path, split="train", debug_mode=debug_mode
        )
        val_entries, node_names, rel_names = from_config.get_data_entries(
            config, anno_path=anno_path, split="val", debug_mode=debug_mode
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

    def _common_forward(self, batch, use_ema=False):
        with torch.profiler.record_function("Prepare Batch"):
            model_input, sbj_target, obj_target, rel_target_cpu, param_target_cpu = (
                prepare_batch(batch, self.device)
            )
            rel_target = rel_target_cpu.to(self.device)
            param_target = param_target_cpu.to(self.device)

        with torch.profiler.record_function("Model Forward"):
            if use_ema:
                assert self.ema_model is not None
                model_out = self.ema_model(model_input)
            else:
                model_out = self.model(model_input)
            sbj_out, obj_out, rel_out, rel_param_out = model_out

        with torch.profiler.record_function("Calc Loss"):
            # node loss
            sbj_loss = self.node_criterion(sbj_out, sbj_target)
            obj_loss = self.node_criterion(obj_out, obj_target)
            node_loss = self.node_loss_weight * (sbj_loss + obj_loss)

            # rel class loss
            rel_loss = self.rel_loss_weight * self.rel_criterion(
                rel_out, rel_target, param_target
            )

            # rel param loss
            if self._use_params:
                rel_param_loss = self.param_criterion(
                    rel_param_out, param_target, rel_target
                )
            else:
                rel_param_loss = torch.tensor(0, dtype=torch.float32)

            loss = node_loss + rel_loss + rel_param_loss

        # uncomment the following code block if there are some errors during training and the weights or loss gets NaN
        # img_ids = torch.repeat_interleave(
        #     torch.arange(len(batch["img"])), batch["num_relations"]
        # )
        # with torch.profiler.record_function("Check NaN"):
        #     if torch.isnan(loss):
        #         if self.out_dir is not None:
        #             torch.save(
        #                 {
        #                     "rel_out": rel_out,
        #                     "param_out": rel_param_out,
        #                     "sbj_out": sbj_out,
        #                     "obj_out": obj_out,
        #                     "rel_tgt": rel_target,
        #                     "param_tgt": param_target,
        #                     "sbj_tgt": sbj_target,
        #                     "obj_tgt": obj_target,
        #                     "img_ids": img_ids,
        #                     "node_loss": node_loss,
        #                     "rel_loss": rel_loss,
        #                     "rel_param_loss": rel_param_loss,
        #                 },
        #                 self.out_dir / "dbg-nanloss.pth",
        #             )
        #         raise RuntimeError("NaN loss")

        with torch.profiler.record_function("Prepare Dict"):
            return {
                "loss": loss,
                "node_loss": node_loss.detach().cpu(),
                "rel_loss": rel_loss.detach().cpu(),
                "rel_param_loss": rel_param_loss.detach().cpu(),
                "sbj_target": sbj_target,
                "obj_target": obj_target,
                "sbj_out": sbj_out,
                "obj_out": obj_out,
                "rel_target": rel_target_cpu,
                "param_target": param_target_cpu,
                "rel_out": rel_out,
                "param_out": rel_param_out,
            }

    @torch.inference_mode()
    def evaluate(
        self,
        epoch: int,
        use_ema=False,
        psg_depth_eval_hack=False,
        profiler=None,
        max_batches=None,
    ):
        if use_ema:
            self.ema_model.eval()
        else:
            self.model.eval()

        node_losses = []
        rel_losses = []
        param_losses = []
        final_losses = []

        grouped_instance_targets = defaultdict(list)
        grouped_instance_outputs = defaultdict(list)
        grouped_pairs = defaultdict(list)

        all_rel_targets = defaultdict(list)
        all_rel_outputs = defaultdict(list)
        all_param_targets = defaultdict(list)
        all_param_outputs = defaultdict(list)

        batch_iterator = tqdm(
            self.val_loader,
            leave=False,
            desc="eval",
            dynamic_ncols=True,
            disable=self.hide_batch_progress,
        )
        for bi, batch in enumerate(batch_iterator):
            fwd = self._common_forward(batch, use_ema=use_ema)

            # instance output
            instance_targets_sbjobj = torch.stack(
                (
                    fwd["sbj_target"].cpu().clone(),
                    fwd["obj_target"].cpu().clone(),
                ),
                dim=-1,
            )
            instance_outputs_sbjobj = torch.stack(
                (
                    fwd["sbj_out"].argmax(dim=-1).cpu().clone(),
                    fwd["obj_out"].argmax(dim=-1).cpu().clone(),
                ),
                dim=-1,
            )

            # rel output
            img_ids = torch.repeat_interleave(
                torch.arange(len(batch["img"])), batch["num_relations"]
            )
            cpu_rel_target = fwd["rel_target"].cpu().clone()
            cpu_rel_out = fwd["rel_out"].sigmoid().cpu().clone()
            cpu_param_target = fwd["param_target"].cpu().clone()
            cpu_param_out = fwd["param_out"].sigmoid().cpu().clone()
            cpu_pair_ids = batch["pairs"].cpu().clone()
            for i, raw_img_id in enumerate(batch["image_id"].tolist()):
                all_rel_targets[raw_img_id].append(cpu_rel_target[img_ids == i])
                all_rel_outputs[raw_img_id].append(cpu_rel_out[img_ids == i])
                all_param_targets[raw_img_id].append(cpu_param_target[img_ids == i])
                all_param_outputs[raw_img_id].append(cpu_param_out[img_ids == i])

                # grouped instances
                grouped_instance_targets[raw_img_id].append(
                    instance_targets_sbjobj[img_ids == i]
                )
                grouped_instance_outputs[raw_img_id].append(
                    instance_outputs_sbjobj[img_ids == i]
                )
                grouped_pairs[raw_img_id].append(cpu_pair_ids[img_ids == i])

            final_losses.append(fwd["loss"].cpu().clone())
            node_losses.append(fwd["node_loss"])
            rel_losses.append(fwd["rel_loss"])
            param_losses.append(fwd["rel_param_loss"])

            if profiler is not None:
                profiler.step()
            if max_batches is not None and bi >= max_batches:
                return

        def _merge(keys, d):
            out = []
            for k in keys:
                v = d[k]
                if len(v) == 1:
                    out.append(v[0])
                else:
                    out.append(torch.cat(v))
            return out

        # using the same keys for all _merge() calls to ensure the same order of items
        all_img_ids = list(all_rel_targets.keys())
        all_rel_targets = _merge(all_img_ids, all_rel_targets)
        all_rel_outputs = _merge(all_img_ids, all_rel_outputs)
        all_param_targets = _merge(all_img_ids, all_param_targets)
        all_param_outputs = _merge(all_img_ids, all_param_outputs)
        grouped_instance_targets = _merge(all_img_ids, grouped_instance_targets)
        grouped_instance_outputs = _merge(all_img_ids, grouped_instance_outputs)
        grouped_pairs = _merge(all_img_ids, grouped_pairs)

        per_class_instance_acc = confusion_matrix(
            # targets
            torch.cat(grouped_instance_targets).flatten(),
            # outputs
            torch.cat(grouped_instance_outputs).flatten(),
            normalize="true",
        ).diagonal()

        metrics = {
            "epoch_loss/val/sum": torch.tensor(final_losses).mean().item(),
            "epoch_loss/val/node": torch.tensor(node_losses).mean().item(),
            "epoch_loss/val/rel": torch.tensor(rel_losses).mean().item(),
            "epoch_loss/val/param": torch.tensor(param_losses).mean().item(),
            "node_acc/mean": float(per_class_instance_acc.mean()),
        }

        metrics.update(
            build_rel_metrics_dict(
                rel_names=self.val_loader.dataset.rel_names[1:],
                rel_thresher=self.rel_thresher,
                gt_list=all_rel_targets,
                output_list=all_rel_outputs,
                param_gt_list=all_param_targets,
                nan_ok=self._debug_mode or psg_depth_eval_hack or self._hack_vg_nan_ok,
            )
        )

        if self._use_params:
            metrics.update(
                build_isg_metrics_dict(
                    rel_names=self.val_loader.dataset.rel_names,
                    rel_thresher=self.rel_thresher,
                    gt_list=all_rel_targets,
                    output_list=all_rel_outputs,
                    param_gt_list=all_param_targets,
                    param_output_list=all_param_outputs,
                )
            )

        for name, acc in zip(
            self.val_loader.dataset.node_names, per_class_instance_acc
        ):
            metrics[f"node_class_acc/{name}"] = acc

        data = {
            "instance_targets": grouped_instance_targets,
            "instance_outputs": grouped_instance_outputs,
            "rel_targets": all_rel_targets,
            "rel_outputs": all_rel_outputs,
            "param_targets": all_param_targets,
            "param_outputs": all_param_outputs,
            "rel_img_ids": all_img_ids,
            "pair_ids": grouped_pairs,
        }

        return metrics, data

    def train_one_epoch(self, epoch: int, profiler=None, max_batches=None):
        self.model.train()

        batch_iterator = tqdm(
            self.train_loader,
            leave=False,
            desc="train",
            dynamic_ncols=True,
            disable=self.hide_batch_progress,
        )

        self.optimizer.zero_grad()
        target_class_counter = torch.zeros(
            len(self.train_loader.dataset.rel_names), dtype=torch.float
        )
        n_rel = 0
        running_loss = {"sum": 0, "node": 0, "rel": 0, "rel_param": 0}
        loss_every = 100
        for bi, batch in enumerate(batch_iterator):
            fwd = self._common_forward(batch)

            # count stats
            target_class_counter += fwd["rel_target"].cpu().sum(0)
            n_rel += len(fwd["rel_target"])

            loss = fwd["loss"] / self.grad_accumulate
            loss.backward()

            # gradient clipping
            if self.grad_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clipping
                )

            if (bi + 1) % self.grad_accumulate == 0:
                self.optimizer.step()
                if self.ema_model is not None:
                    self.ema_model.update_parameters(self.model)
                self.optimizer.zero_grad()

            running_loss["sum"] += fwd["loss"].detach().cpu().item()
            for sub in ("node", "rel", "rel_param"):
                running_loss[sub] += fwd[f"{sub}_loss"].detach().cpu().item()

            # don't clutter the tensorboard and only report every 100th batch
            if (bi + 1) % loss_every == 0:
                for sub in ("sum", "node", "rel", "rel_param"):
                    self.tensorboard.add_scalar(
                        f"loss/train/{sub}",
                        running_loss[sub] / loss_every,
                        global_step=self._global_step,
                    )
                running_loss = {"sum": 0, "node": 0, "rel": 0, "rel_param": 0}

            self._global_step += 1

            if profiler is not None:
                profiler.step()
            if max_batches is not None and bi >= max_batches:
                break

        torch.save(
            {
                "counts": target_class_counter.tolist(),
                "n_rel": int(n_rel),
                "pos_weight": self.rel_criterion.pos_weight.cpu().tolist(),
            },
            self.out_dir / "train_stats.pth",
        )

        # don't really know if this helps with reducing VRAM usage but it won't harm I guess
        del (batch, loss, fwd)

    def run(self, epochs, start_epoch=0):
        for epoch in tqdm(range(start_epoch, epochs)):
            self.one_epoch(epoch)

    def one_epoch(self, epoch):
        self.train_one_epoch(epoch)

        metrics, out_data = self.evaluate(epoch)
        for key, value in metrics.items():
            self.tensorboard.add_scalar(key, value, global_step=epoch)

        crit_value = metrics[self.critical_metric]

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            # show learning rate during training
            for i, pg in enumerate(self.optimizer.param_groups):
                self.tensorboard.add_scalar(f"_lr/{i}", pg["lr"], global_step=epoch)

        is_best_epoch = False
        if self.best_metric_value is None or crit_value > self.best_metric_value:
            self.best_metric_value = crit_value
            is_best_epoch = True

        if self.out_dir and self.dump_data:
            model_state = {
                "epoch": epoch,
                "_global_step": self._global_step,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "lr_scheduler": (
                    None
                    if self.lr_scheduler is None
                    else self.lr_scheduler.state_dict()
                ),
                "metric": crit_value,
            }
            if self.ema_model is not None:
                model_state["ema_model"] = self.ema_model.state_dict()

            torch.save(model_state, self.out_dir / "last_state.pth")
            torch.save(out_data, self.out_dir / "last_data.pth")

            if is_best_epoch:
                torch.save(model_state, self.out_dir / "best_state.pth")
                torch.save(out_data, self.out_dir / "best_data.pth")
                with open(self.out_dir / "best_metrics.json", "w") as f:
                    json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

        # run an EMA evaluation
        if self.ema_model is not None:
            ema_metrics, _ = self.evaluate(epoch, use_ema=True)
            for key, value in ema_metrics.items():
                self.tensorboard.add_scalar(key + "_EMA", value, global_step=epoch)

        return metrics
