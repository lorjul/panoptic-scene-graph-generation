# this file contains methods that read a config object and return a structure based on the read config
# this file exists such that config.py has no dependencies on the rest of the code
# and such that no other low-level file has a dependency on config.py
from pathlib import Path
import torch
import torch.optim.lr_scheduler as lrs
from .config import (
    Config,
    ExtractorCfg_ConvNeXt,
    ExtractorCfg_Dinov2,
    ExtractorCfg_Dinov2Single,
    ExtractorCfg_Dinov3,
    ExtractorCfg_EoMT,
    ExtractorCfg_EomtHf,
    ExtractorCfg_FasterRCNN,
    ExtractorCfg_HRNet,
    ExtractorCfg_Mask2Former,
    ExtractorCfg_Radio,
    ExtractorCfg_ResNet,
    LRScheduleStep,
)
from .models.feature_extractors import (
    FasterRCNNExtractor,
    Mask2FormerExtractor,
    MergedMask2FormerExtractor,
    FullMask2FormerExtractor,
    GenericTimmExtractor,
    Dinov2Extractor,
    Dinov2SingleExtractor,
    Dinov3Extractor,
    EoMTExtractor,
    EomtHfExtractor,
    RadioExtractor,
    ResNetExtractor,
)
from .models.daniformer import DaniFormer
from .data.augmentation import config_to_aug, get_standard_transforms
from .data.load_entries import (
    load_psg_entries,
    load_objects365_entries,
    load_psgcoco_entries,
    load_visual_genome_entries,
    load_isg_entries,
)
from sgadl.utils import get_device
from sgadl.relparam import TargetThresholder


def get_extractor(extractor_cfg):
    if isinstance(extractor_cfg, ExtractorCfg_FasterRCNN):
        return FasterRCNNExtractor(
            feature_key=extractor_cfg.stage, architecture=extractor_cfg.backbone
        ).requires_grad_(not extractor_cfg.frozen)
    elif isinstance(extractor_cfg, ExtractorCfg_ResNet):
        return ResNetExtractor(name=extractor_cfg.name)
    elif isinstance(extractor_cfg, ExtractorCfg_Mask2Former):
        if extractor_cfg.features == "encoder":
            return Mask2FormerExtractor(checkpoint_name=extractor_cfg.checkpoint)
        elif extractor_cfg.features == "decoder":
            return FullMask2FormerExtractor(checkpoint_name=extractor_cfg.checkpoint)
        elif extractor_cfg.features == "decoder_merged":
            return MergedMask2FormerExtractor(checkpoint_name=extractor_cfg.checkpoint)
        else:
            raise KeyError(extractor_cfg.sub_type)
    elif isinstance(extractor_cfg, ExtractorCfg_HRNet):
        return GenericTimmExtractor(
            model_name=f"hrnet_{extractor_cfg.variant}", feature_index=1
        )
    elif isinstance(extractor_cfg, ExtractorCfg_ConvNeXt):
        return GenericTimmExtractor(
            model_name=f"convnext_{extractor_cfg.variant}", feature_index=1
        )
    elif isinstance(extractor_cfg, ExtractorCfg_Dinov2):
        # DINOv2 is set to frozen, contrary to the other models that we have
        return Dinov2Extractor(
            variant=extractor_cfg.variant, num_layers=extractor_cfg.num_intermediate
        ).requires_grad_(not extractor_cfg.frozen)
    elif isinstance(extractor_cfg, ExtractorCfg_Dinov2Single):
        # DINOv2 is set to frozen, contrary to the other models that we have
        return Dinov2SingleExtractor(
            variant=extractor_cfg.variant, layer_idx=extractor_cfg.layer_idx
        ).requires_grad_(not extractor_cfg.frozen)
    elif isinstance(extractor_cfg, ExtractorCfg_Dinov3):
        return Dinov3Extractor(
            variant=extractor_cfg.variant, multi_layers=extractor_cfg.multi_layers
        ).requires_grad_(not extractor_cfg.frozen)
    elif isinstance(extractor_cfg, ExtractorCfg_Radio):
        return RadioExtractor(model_version=extractor_cfg.variant).requires_grad_(
            not extractor_cfg.frozen
        )
    elif isinstance(extractor_cfg, ExtractorCfg_EoMT):
        return EoMTExtractor(
            variant=extractor_cfg.variant,
            img_size=extractor_cfg.img_size,
            num_layers=extractor_cfg.num_intermediate,
        ).requires_grad_(not extractor_cfg.frozen)
    elif isinstance(extractor_cfg, ExtractorCfg_EomtHf):
        return EomtHfExtractor(
            model_id=extractor_cfg.model_id,
            num_layers=extractor_cfg.num_intermediate,
        ).requires_grad_(not extractor_cfg.frozen)
    raise KeyError(extractor_cfg)


def get_model(config: Config, num_node_outputs: int, num_rel_outputs: int):
    arch_cfg = config.architecture
    extractor = get_extractor(config.extractor).to(get_device())
    # TODO: don't hardcode width and height
    if isinstance(config.extractor, ExtractorCfg_Dinov2):
        feature_shape = extractor.get_feature_shape((3, 644, 644))
    else:
        feature_shape = extractor.get_feature_shape((3, 640, 640))

    model = DaniFormer(
        num_node_outputs=num_node_outputs,
        num_rel_outputs=num_rel_outputs,
        extractor=extractor,
        transformer_depth=arch_cfg.transformer_depth,
        embed_dim=arch_cfg.embed_dim,
        patch_size=arch_cfg.patch_size,
        feature_shape=feature_shape,
        use_semantics=arch_cfg.use_semantics,
        use_masks=arch_cfg.use_masks,
        bg_ratio_strategy=arch_cfg.bg_ratio_strategy,
        encode_coords=arch_cfg.encode_coords,
    )
    if config.compile:
        return torch.compile(model)
    return model


def get_ema_model(config: Config, base_model: torch.nn.Module, device=None):
    if config.ema is None:
        return None

    return torch.optim.swa_utils.AveragedModel(
        model=base_model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema),
        # apparently, we could also recalculate the buffers at the end after training
        # but that's probably going to be a bit more of a hassle than just setting use_buffers to True
        use_buffers=True,
        device=device,
    )


def get_augmentations(config: Config, split: str):
    if isinstance(config.extractor, ExtractorCfg_Dinov2):
        img_size = 644
    else:
        img_size = 640

    if config is None or config.augmentations is None:
        return get_standard_transforms(is_train=split == "train", image_size=img_size)
    return [config_to_aug(c) for c in config.augmentations.get_list(split)]


def get_data_entries(config: Config, anno_path, split, img_dir=None, debug_mode=False):
    if config.data.source == "psg":
        return load_psg_entries(
            anno_path=anno_path, split=split, max_entries=10 if debug_mode else None
        )
    if config.data.source == "o365":
        assert split in ("train", "val")
        assert isinstance(img_dir, (Path, str))
        return load_objects365_entries(
            anno_path=f"{anno_path}{split}.json", img_dir=Path(img_dir) / split
        )
    if config.data.source == "psg-coco":
        assert split in ("train", "val")
        anno_path = Path(anno_path)
        return load_psgcoco_entries(
            anno_path=anno_path / "annotations" / f"panoptic_{split}2017.json",
            img_prefix=f"{split}2017",
            seg_prefix=f"panoptic_{split}2017",
        )
    if config.data.source == "vg-ietrans":
        anno_path = Path(anno_path)
        assert anno_path.is_dir()
        return load_visual_genome_entries(
            img_data_path=anno_path / "image_data.json",
            # we don't use the 1000 classes
            meta_path=anno_path / "50" / "VG-SGG-dicts-with-attri.json",
            h5_path=anno_path / "50" / "VG-SGG-with-attri.h5",
            split=split,
            variant="ietrans",
            max_entries=50 if debug_mode else None,
        )
    if config.data.source == "vg-danfei":
        anno_path = Path(anno_path)
        assert anno_path.is_dir()
        return load_visual_genome_entries(
            img_data_path=anno_path / "image_data.json",
            meta_path=anno_path / "VG-SGG-dicts.json",
            h5_path=anno_path / "VG-SGG.h5",
            split=split,
            variant="danfei",
            max_entries=50 if debug_mode else None,
        )
    if config.data.source == "vg-kaihua":
        anno_path = Path(anno_path)
        assert anno_path.is_dir()
        return load_visual_genome_entries(
            img_data_path=anno_path / "image_data.json",
            meta_path=anno_path / "VG-SGG-dicts-with-attri.json",
            h5_path=anno_path / "VG-SGG-with-attri.h5",
            split=split,
            variant="kaihua",
            max_entries=50 if debug_mode else None,
        )
    if config.data.source == "isg":
        anno_path = Path(anno_path)
        return load_isg_entries(
            anno_path=anno_path,
            split=split,
            debug_mode=debug_mode,
            ignore_nodes=config.data.ignore_nodes,
        )

    raise ValueError(f"Unsupported dataset: {config.data.source}")


def get_lr_scheduler(config: Config, optimizer):
    if config.lr_schedule is None:
        return None
    if isinstance(config.lr_schedule, LRScheduleStep):
        return lrs.StepLR(
            optimizer=optimizer,
            step_size=config.lr_schedule.step_size,
            gamma=config.lr_schedule.factor,
        )
    raise ValueError("Unsupported lr_schedule config")


def get_rel_thresher(config: Config, relnames):
    if config.rel_thresholds is None:
        return None
    else:
        return TargetThresholder(
            relnames=relnames,
            min_ign_angle_deg=config.rel_thresholds.min_ign_angle_deg,
            max_ign_angle_deg=config.rel_thresholds.max_ign_angle_deg,
            min_ign_distance_m=config.rel_thresholds.min_ign_distance_m,
            max_ign_distance_m=config.rel_thresholds.max_ign_distance_m,
        )
