import json
from typing import Literal, Optional, Sequence, Union
from pydantic import BaseModel, Field, ConfigDict


class ExtractorCfg_FasterRCNN(BaseModel):
    type: Literal["fasterrcnn"] = "fasterrcnn"
    stage: Literal["0", "1", "2", "3", "all"] = "0"
    backbone: Literal["resnet50", "mobilenet"] = "resnet50"
    frozen: bool = False


class ExtractorCfg_ResNet(BaseModel):
    type: Literal["resnet"] = "resnet"
    name: str = "resnet101"


class ExtractorCfg_Mask2Former(BaseModel):
    type: Literal["mask2former"] = "mask2former"
    checkpoint: str = "facebook/mask2former-swin-base-coco-panoptic"
    features: Literal["encoder", "decoder", "decoder_merged"] = "encoder"


class ExtractorCfg_Dinov2(BaseModel):
    type: Literal["dinov2"] = "dinov2"
    variant: Literal["s", "b", "l", "g"] = "s"
    num_intermediate: int = 4
    frozen: bool = True


class ExtractorCfg_Dinov2Single(BaseModel):
    type: Literal["dinov2_single"] = "dinov2_single"
    variant: Literal["s", "b", "l", "g"] = "s"
    layer_idx: int = 4
    frozen: bool = True


class ExtractorCfg_Dinov3(BaseModel):
    type: Literal["dinov3"] = "dinov3"
    variant: Literal["vits16", "vitb16", "vitl16"] = "vits16"
    frozen: bool = True
    multi_layers: Optional[int] = None


class ExtractorCfg_Radio(BaseModel):
    type: Literal["radio"] = "radio"
    variant: Literal["c-radio_v3-b", "c-radio_v3-l", "e-radio_v2"] = "c-radio_v3-b"
    frozen: bool = True


class ExtractorCfg_EoMT(BaseModel):
    type: Literal["eomt"] = "eomt"
    variant: Literal["small", "base", "large", "giant"] = "base"
    img_size: int = 640
    num_intermediate: int = 4
    frozen: bool = True


class ExtractorCfg_EomtHf(BaseModel):
    type: Literal["eomt-hf"] = "eomt-hf"
    model_id: str = "tue-mps/coco_panoptic_eomt_base_640_2x"
    frozen: bool = True
    num_intermediate: int = 4


class ExtractorCfg_HRNet(BaseModel):
    type: Literal["hrnet"] = "hrnet"
    variant: Literal["w32"] = "w32"


class ExtractorCfg_ConvNeXt(BaseModel):
    type: Literal["convnext"] = "convnext"
    variant: str = "large.fb_in22k_ft_in1k_384"


class _Augmentations(BaseModel):
    train: Sequence[Union[dict, str]]
    val: Sequence[Union[dict, str]]

    def get_list(self, split: str):
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        if split == "test":
            return self.val
        raise KeyError(split)


class ArchCfg_DaniFormer(BaseModel):
    transformer_depth: int = 6
    embed_dim: int = 384
    patch_size: int = 8
    use_semantics: bool = False
    use_masks: bool = False
    bg_ratio_strategy: Literal["sum", "onoff"] = "sum"
    encode_coords: bool = False


class DataCfg(BaseModel):
    source: Literal[
        "isg",
        "o365",
        "psg",
        "psg-coco",
        "vg-danfei",
        "vg-ietrans",
        "vg-kaihua",
    ] = "psg"
    "The dataset kind. The vg-* options are for different variants of the VG-150 dataset."

    normals: bool = False
    depth: bool = False
    ignore_nodes: Optional[Sequence[str]] = None


class LRScheduleStep(BaseModel):
    type: Literal["step"] = "step"
    step_size: int = 10
    factor: float = 0.1


class LossCfg(BaseModel):
    rel_weight: float = 0.8
    node_loss_weight: float = 0.2
    angle_weight: float = 0.0
    distance_weight: float = 0.0
    legacy_balancing: bool = False


def _dict_to_markdown(d: dict, indent=0):
    lines = []
    indent_str = " " * indent
    for key, value in d.items():
        if isinstance(value, BaseModel):
            lines.append(f"{indent_str}- {key}:")
            lines.extend(_dict_to_markdown(value.__dict__, indent=indent + 4))
        else:
            lines.append(f"{indent_str}- {key}: {value}")
    return lines


class RelTreshCfg(BaseModel):
    min_ign_angle_deg: float = 10
    max_ign_angle_deg: float = 20
    min_ign_distance_m: float = 1.0
    max_ign_distance_m: float = 1.2


class Config(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    description: Optional[str] = None
    "Description is just to document the used configuration. It will not change the behaviour of the code."

    loss: LossCfg = LossCfg()
    ignore_norel_predicates: bool = False
    rels_per_batch: int = 4096
    lr: float = 0.001

    lr_backbone: Optional[float] = None
    "Set to None to use same lr as config.lr"

    lr_schedule: Union[LRScheduleStep, None] = None
    grad_clipping: Optional[float] = None
    weight_decay: float = 0.01
    neg_ratio: float = 1.0
    augmentations: Optional[_Augmentations] = None
    grad_accumulate: int = 1

    ema: Optional[float] = None
    """If set to a float value, also train an EMA-model with the provided factor. Usually set to 0.995 or higher."""

    rel_thresholds: Optional[RelTreshCfg] = RelTreshCfg()

    extractor: Union[
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
    ] = Field(ExtractorCfg_FasterRCNN(), discriminator="type")
    architecture: ArchCfg_DaniFormer = ArchCfg_DaniFormer()

    compile: bool = False
    "Whether to use torch.compile on the model."

    data: DataCfg = DataCfg()

    def to_markdown(self):
        return "\n".join(_dict_to_markdown(self.__dict__))

    @staticmethod
    def from_file(path):
        with open(path) as f:
            data = json.load(f)
        return Config.model_validate(data, strict=True)

    def to_file(self, path):
        content = self.model_dump()
        with open(path, "w") as f:
            json.dump(content, f, indent=2)

    def with_overrides(self, overrides: dict):
        curdict = self.model_dump()

        for key, value in overrides.items():
            keys = key.split(".")
            cur = curdict
            for k in keys[:-1]:
                cur = cur[k]
            if keys[-1] not in cur:
                raise RuntimeError(f"Unknown config key: {key}")
            cur[keys[-1]] = value

        return Config.model_validate(curdict, strict=False)


def _write_schema():
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    schema_str = Config.model_json_schema()
    with open(args.output, "w") as f:
        json.dump(schema_str, f, indent=2)


if __name__ == "__main__":
    _write_schema()
