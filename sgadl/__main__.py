from typing import Union, Optional
from pathlib import Path
from argparse import ArgumentParser
import json
from project_paths import project_paths
import torch
import submitit
from .config import Config
from .trainer import Trainer
from .utils import detect_task_spooler
from .profiler_wrapper import get_profiler


def run_training(
    config_path: str,
    model_path: Union[None, str],
    no_bpbar: bool,
    num_epochs: int,
    anno_path: Optional[str],
    img_dir: Optional[str],
    seg_dir: Optional[str],
    out_dir: str,
    num_workers: int,
    debug_mode: bool,
    is_transfer_learning: bool,
    use_profiler: bool,
    cfg_overrides: list,
    args_dict: dict,
):
    config = Config.from_file(config_path)

    # override some config keys via CLI
    if cfg_overrides:
        overrides = {}
        for c in cfg_overrides:
            k, v = c.split("=")
            overrides[k] = v
        config = config.with_overrides(overrides)

    # replace anno_path, img_dir, seg_dir with values from project_paths if they have not been provided by the user
    if anno_path is None:
        anno_path = project_paths.get_anno_path(config.data.source)
    if img_dir is None:
        img_dir = project_paths.get_img_dir(config.data.source)
    if seg_dir is None:
        seg_dir = project_paths.get_seg_dir(config.data.source)

    if model_path is None:
        state_dict = None
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        assert "_global_step" in state_dict

    if no_bpbar:
        hide_batch_progress = True
    elif detect_task_spooler():
        print("Detected task-spooler. Batch progress will be hidden")
        hide_batch_progress = True
    else:
        hide_batch_progress = False

    trainer = Trainer(
        anno_path=anno_path,
        img_dir=img_dir,
        seg_dir=seg_dir,
        out_dir=out_dir,
        config=config,
        num_workers=num_workers,
        start_state_dict=state_dict,
        hide_batch_progress=hide_batch_progress,
        debug_mode=debug_mode,
        is_transfer_learning=is_transfer_learning,
    )

    if trainer.out_dir:
        with open(trainer.out_dir / "args.json", "w") as f:
            json.dump(args_dict, f)

    if use_profiler:
        wait = 1
        warmup = 2
        active = 3
        repeat = 2
        max_steps = wait + (1 + repeat) * (warmup + active)
        with get_profiler(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            log_dir=Path(out_dir) / "train",
        ) as prof:
            trainer.train_one_epoch(epoch=0, profiler=prof, max_batches=max_steps + 1)
        perf = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=15
        )
        with open(Path(out_dir) / "train_profiler.txt", "w") as f:
            f.write(perf)
        # prof.export_chrome_trace(str(Path(out_dir) / "train.chrome_trace.json"))

        with get_profiler(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            log_dir=Path(out_dir) / "val",
        ) as prof:
            trainer.evaluate(epoch=0, profiler=prof, max_batches=max_steps + 1)
            perf = prof.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=15
            )
        with open(Path(out_dir) / "eval_profiler.txt", "w") as f:
            f.write(perf)
        # prof.export_chrome_trace(str(Path(out_dir) / "eval.chrome_trace.json"))
    else:
        trainer.run(epochs=num_epochs)

    # to easily check which experiments ran to the end
    if trainer.out_dir:
        with open(trainer.out_dir / "done.txt", "w") as f:
            f.write("done")

    return trainer


def cli():
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "output",
        help="Path to output folder. Note that all contents inside that folder will be overwritten",
    )
    # TODO: update help str for non-PSG datasets
    parser.add_argument("--anno", default=None, help="Path to PSG JSON annotation file")
    parser.add_argument("--img", default=None, help="Directory that contains images")
    parser.add_argument(
        "--seg", default=None, help="Directory that contains segmentation masks"
    )
    parser.add_argument(
        "--epochs",
        default=40,
        type=int,
        help="Total number of training epochs. If resuming, will run until the desired number was reached in total.",
    )
    parser.add_argument(
        "--workers", default=8, type=int, help="Number of workers for the data loaders"
    )
    parser.add_argument(
        "--continue-from", default=None, help="Resume from this checkpoint file."
    )
    parser.add_argument(
        "--transfer-state",
        default=None,
        help="Use this checkpoint file as a start for transfer learning.",
    )
    parser.add_argument("--no-bpbar", default=False, action="store_true")
    parser.add_argument(
        "--slurm",
        default=None,
        type=str,
        help="Path to Slurm config if the training should be run using Slurm. The config file must be a JSON file that contains the kwargs for executor.update_parameters().",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="In debug mode, only a small subset of the data will be used.",
    )
    parser.add_argument("--path-eomt", default=None)
    parser.add_argument("--dinov3-weights", default=None)
    parser.add_argument(
        "--profiler",
        default=False,
        action="store_true",
        help="Whether to run the PyTorch profiler. WARNING: Somehow it crashes with --workers 4. --workers 16 seems to work.",
    )
    parser.add_argument(
        "--cfg",
        nargs="+",
        help="""Overrides for the config file. Must be specified as key=value pairs.
You can list multiple entries like --cfg architecture.transformer_depth=3 data.source=psg""",
    )
    args = parser.parse_args()

    assert (
        args.transfer_state is None or args.continue_from is None
    ), "You cannot specify --transfer-state and --continue-from at the same time."
    if args.transfer_state:
        model_path = args.transfer_state
        is_transfer_learning = True
    elif args.continue_from:
        model_path = args.continue_from
        is_transfer_learning = False
    else:
        model_path = None
        is_transfer_learning = False

    if args.path_eomt:
        project_paths.eomt_weights = Path(args.path_eomt)
    if args.dinov3_weights:
        project_paths.dinov3_weights = Path(args.dinov3_weights)

    # kwargs for run_training
    # will be used for either submitit or direct execution
    kwargs = dict(
        config_path=args.config,
        model_path=model_path,
        no_bpbar=args.no_bpbar,
        num_epochs=args.epochs,
        anno_path=args.anno,
        img_dir=args.img,
        seg_dir=args.seg,
        out_dir=args.output,
        num_workers=args.workers,
        debug_mode=args.debug,
        is_transfer_learning=is_transfer_learning,
        use_profiler=args.profiler,
        cfg_overrides=args.cfg,
        args_dict=args.__dict__,
    )

    if args.slurm:
        with open(args.slurm) as f:
            slurm_cfg = json.load(f)

        out_dir = Path(args.output)
        out_dir.mkdir(exist_ok=True, parents=True)

        executor = submitit.AutoExecutor(folder=out_dir / "slurm")
        executor.update_parameters(slurm_job_name=f"mlbl-{out_dir.name}", **slurm_cfg)

        job = executor.submit(run_training, **kwargs)
        print("Scheduled job:", job.job_id)
    else:
        trainer = run_training(**kwargs)

        print(
            "Best value for ",
            trainer.critical_metric,
            ": ",
            trainer.best_metric_value,
            sep="",
        )


if __name__ == "__main__":
    cli()
