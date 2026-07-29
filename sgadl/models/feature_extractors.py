# contains different feature extractors
from typing import Literal, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

try:
    from transformers import Mask2FormerForUniversalSegmentation
    from transformers.models.eomt.image_processing_eomt_fast import (
        EomtImageProcessorFast,
    )
    from transformers.models.eomt.modeling_eomt import EomtForUniversalSegmentation
except ImportError:
    print("transformers is not available")
try:
    import timm
except ImportError:
    print("timm is not available")

from project_paths import project_paths
from sgadl.utils import get_device
from sgadl.models.eomt import make_eomt, download_eomt_ckpt


class FeatureExtractor(nn.Module):
    def get_feature_shape(self, input_shape):
        with torch.inference_mode():
            # move the tensor to the GPU, some efficient implementations require CUDA for forward passes
            dummy_tensor = torch.empty((1, *input_shape), device=get_device())
            tmp = self.forward(dummy_tensor)
        return tmp.shape[1:]

    def forward(self, img: torch.Tensor):
        raise NotImplementedError()


class FasterRCNNExtractor(FeatureExtractor):
    def __init__(self, feature_key="0", architecture="resnet50"):
        super().__init__()
        if architecture == "resnet50":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            ).backbone
        elif architecture == "mobilenet":
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            ).backbone
        else:
            raise ValueError("Unknown architecture")
        self.feature_key = feature_key

        if feature_key == "all":
            self.feature_squasher = nn.Conv2d(
                in_channels=1024, out_channels=256, kernel_size=1
            )

    def forward(self, img: torch.Tensor):
        features = self.model(img)

        if self.feature_key == "all":
            # resize all feature maps to the largest size
            target_shape = features["0"].shape[-2:]
            combined = torch.cat(
                [
                    features["0"],
                    F.interpolate(features["1"], target_shape),
                    F.interpolate(features["2"], target_shape),
                    F.interpolate(features["3"], target_shape),
                ],
                dim=1,
            )

            return self.feature_squasher(combined)

        return features[self.feature_key]


class ResNetExtractor(FeatureExtractor):
    def __init__(self, name: str, feature_key="0"):
        super().__init__()
        weights = {
            "resnet101": torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V2,
            "resnet152": torchvision.models.resnet.ResNet152_Weights.IMAGENET1K_V2,
            "resnet50": torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2,
            "resnet18": torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": torchvision.models.resnet.ResNet34_Weights.IMAGENET1K_V1,
            "resnext50": torchvision.models.resnet.ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
            "resnext101_64": torchvision.models.resnet.ResNeXt101_64X4D_Weights.IMAGENET1K_V1,
            "resnext101_32": torchvision.models.resnet.ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
        }
        self.model = resnet_fpn_backbone(backbone_name=name, weights=weights[name])
        self.feature_key = feature_key

    def forward(self, img: torch.Tensor):
        features = self.model(img)
        return features[self.feature_key]


class Mask2FormerExtractor(FeatureExtractor):
    def __init__(
        self,
        checkpoint_name="facebook/mask2former-swin-base-coco-panoptic",
        feature_key=1,
    ):
        super().__init__()
        mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint_name
        )
        self.model = mask2former.model.pixel_level_module.encoder
        self.feature_key = feature_key

    def forward(self, img: torch.Tensor):
        features = self.model(img).feature_maps
        return features[self.feature_key]


class FullMask2FormerExtractor(FeatureExtractor):
    def __init__(self, checkpoint_name="facebook/mask2former-swin-base-coco-panoptic"):
        super().__init__()
        mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint_name
        )
        self.model = mask2former.model.pixel_level_module

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img).decoder_last_hidden_state


class MergedMask2FormerExtractor(FeatureExtractor):
    def __init__(self, checkpoint_name="facebook/mask2former-swin-base-coco-panoptic"):
        super().__init__()
        mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint_name
        )
        self.model = mask2former.model.pixel_level_module

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        states = self.model(img).decoder_hidden_states
        target_size = states[-1].shape[-2:]
        reshaped = [F.interpolate(t, target_size) for t in states]
        return torch.stack(reshaped, dim=0).sum(dim=0)


class Dinov2Extractor(FeatureExtractor):
    """Extractor using DINOv2: https://github.com/facebookresearch/dinov2
    This extractor requires the images to be normalized using ImageNet statistics.
    This extractor differs from Dinov2SingleExtractor by using fusing multiple intermediate
    layers to return the feature output.
    """

    def __init__(self, variant: Literal["s", "b", "l", "g"] = "s", num_layers: int = 4):
        super().__init__()
        # patch size is always 14 for DINOv2
        self.patch_size = 14
        # load DINOv2 from Torch Hub
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2", f"dinov2_vit{variant}{self.patch_size}_reg"
            )
        except RuntimeError:
            # try again, but this time with force_reload in case the cache is not up to date
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                f"dinov2_vit{variant}{self.patch_size}_reg",
                force_reload=True,
            )

        # which intermediate layers to choose?
        layer_step = len(self.model.blocks) // num_layers
        self.extract_layers = list(range(len(self.model.blocks) - 1, -1, -layer_step))
        self.extract_layers = self.extract_layers[::-1]
        # alternate way, which is a bit cleaner: (is it identical?)
        assert set(self.extract_layers) == set(
            range(layer_step - 1, len(self.model.blocks), layer_step)
        )
        assert len(self.extract_layers) == num_layers
        # a merge layer that will merge the intermediate features into one single tensor
        self.merge = nn.Linear(in_features=len(self.extract_layers), out_features=1)

    def forward(self, img: torch.Tensor):
        tokens = self.model.get_intermediate_layers(img, n=self.extract_layers)
        tokens = torch.stack(tokens, dim=-1).transpose(1, 2)
        features = self.merge(tokens)
        features = features.reshape(
            features.size(0),
            features.size(1),
            img.size(2) // self.patch_size,
            img.size(3) // self.patch_size,
        )
        return features


class Dinov2SingleExtractor(Dinov2Extractor):
    """Extractor using DINOv2: https://github.com/facebookresearch/dinov2
    This extractor requires the images to be normalized using ImageNet statistics.
    This extractor differs from Dinov2Extractor by directly using the output from
    a specific layer as feature output without fusing multiple intermediate layers.
    """

    def __init__(self, variant="s", layer_idx=4):
        super().__init__()
        # patch size is always 14 for DINOv2
        self.patch_size = 14
        # load DINOv2 from Torch Hub
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2", f"dinov2_vit{variant}{self.patch_size}_reg"
            )
        except RuntimeError:
            # try again, but this time with force_reload in case the cache is not up to date
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                f"dinov2_vit{variant}{self.patch_size}_reg",
                force_reload=True,
            )
        self.layer_idx = layer_idx

    def forward(self, img: torch.Tensor):
        tokens = self.model.get_intermediate_layers(img, n=[self.layer_idx])[0]
        features = tokens.transpose(1, 2)
        return features.reshape(
            features.size(0),
            features.size(1),
            img.size(2) // self.patch_size,
            img.size(3) // self.patch_size,
        )


class Dinov3Extractor(FeatureExtractor):
    """Extractor using DINOv3: https://github.com/facebookresearch/dinov3
    This extractor requires the images to be normalized using ImageNet statistics.
    """

    def __init__(
        self, variant="vitb16", multi_layers: Optional[int] = None, pretrained=True
    ):
        super().__init__()

        # from https://github.com/facebookresearch/dinov3/blob/main/notebooks/pca.ipynb
        model2num_layers = {
            "vits16": 12,
            # MODEL_DINOV3_VITSP: 12,
            "vitb16": 12,
            "vitl16": 24,
            # MODEL_DINOV3_VITHP: 32,
            # MODEL_DINOV3_VIT7B: 40,
        }
        assert variant in model2num_layers

        if pretrained:
            # look for a weights file that starts with the correct name
            found_ckpts = list(
                project_paths.dinov3_weights.glob(f"dinov3_{variant}_pretrain*.pth")
            )
            if len(found_ckpts) != 1:
                raise FileNotFoundError(
                    f"Could not identify dinov3 checkpoint for {variant}"
                )
            ckpt_path = str(found_ckpts[0])
            self.model = torch.hub.load(
                str(Path(__file__).parent.parent.parent / "dinov3"),
                f"dinov3_{variant}",
                source="local",
                weights=ckpt_path,
            )
        else:
            self.model = torch.hub.load(
                "facebookresearch/dinov3", f"dinov3_{variant}", pretrained=False
            )

        num_layers = model2num_layers[variant]

        if multi_layers is None:
            self.extract_layers = [num_layers - 1]
            self.merge = None
        else:
            layer_step = num_layers // multi_layers
            self.extract_layers = list(range(layer_step - 1, num_layers, layer_step))
            self.merge = nn.Linear(in_features=len(self.extract_layers), out_features=1)

    def forward(self, img: torch.Tensor):
        tokens = self.model.get_intermediate_layers(
            img, n=self.extract_layers, reshape=True
        )
        if self.merge is None:
            return tokens[0]

        # use more intermediate layers
        features = self.merge(torch.stack(tokens, dim=-1)).squeeze(-1)
        return features


class RadioExtractor(FeatureExtractor):
    def __init__(self, model_version: str):
        super().__init__()
        assert model_version in ("c-radio_v3-b", "c-radio_v3-l", "e-radio_v2")
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=model_version,
            progress=True,
            # skip_validation=True,
            trust_repo=True,
        )
        # preprocessor handles image normalization using the correct weights
        # we do this using the augmentation pipeline instead
        # but watch out to use the correct weights
        _ = self.model.make_preprocessor_external()

    def forward(self, img: torch.Tensor):
        _, ftrs = self.model(img)

        features = ftrs.transpose(1, 2)
        return features.reshape(
            features.size(0),
            features.size(1),
            img.size(2) // self.model.min_resolution_step,
            img.size(3) // self.model.min_resolution_step,
        )


class EoMTExtractor(FeatureExtractor):
    """Extractor using the backend from EoMT: https://github.com/tue-mps/eomt
    This extractor requires the images to be normalized to the interval [0,1].
    EoMT performs internal normalization!
    """

    def __init__(
        self,
        variant: Literal["small", "base", "large", "giant"],
        img_size: int,
        num_layers: int = 4,
        pretrained=True,
    ):
        super().__init__()
        if pretrained:
            ckpt_path = download_eomt_ckpt(
                variant=variant, img_size=img_size, save_dir=project_paths.eomt_weights
            )
        else:
            ckpt_path = None
        eomt = make_eomt(variant=variant, ckpt_path=ckpt_path, img_size=img_size)
        self.encoder = eomt.encoder
        self.patch_size = 16

        # which intermediate layers to choose?
        layer_step = len(self.encoder.backbone.blocks) // num_layers
        self.extract_layers = list(
            range(len(self.encoder.backbone.blocks) - 1, -1, -layer_step)
        )
        self.extract_layers = self.extract_layers[::-1]
        assert len(self.extract_layers) == num_layers
        # a merge layer that will merge the intermediate features into one single tensor
        self.merge = nn.Linear(in_features=len(self.extract_layers), out_features=1)

    def forward(self, img: torch.Tensor):
        # EoMT stores image statistics inside the model
        # images should not be transformed externally but here:
        img = (img - self.encoder.pixel_mean) / self.encoder.pixel_std

        _, tokens = self.encoder.backbone.forward_intermediates(
            img, indices=self.extract_layers
        )
        tokens = torch.stack(tokens, dim=-1)
        features = self.merge(tokens)
        features = features.squeeze(-1)
        return features


class EomtHfExtractor(FeatureExtractor):
    def __init__(self, model_id: str, num_layers: int = 4):
        super().__init__()
        self.processor = EomtImageProcessorFast.from_pretrained(model_id)
        self.model = EomtForUniversalSegmentation.from_pretrained(model_id)

        # which intermediate layers to choose?
        layer_step = self.model.num_hidden_layers // num_layers
        self.extract_layers = list(
            range(self.model.num_hidden_layers - 1, -1, -layer_step)
        )
        self.extract_layers = self.extract_layers[::-1]
        assert len(self.extract_layers) == num_layers
        # a merge layer that will merge the intermediate features into one single tensor
        self.merge = nn.Linear(in_features=len(self.extract_layers), out_features=1)

    def forward(self, img: torch.Tensor, return_segmentation=False):
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)

        token_list = []
        for l in self.extract_layers:
            # the first tokens in hidden_states are reserved for DINOv2's prefix tokens
            # the last num_blocks layers also have EoMT's query tokens
            offset = self.model.embeddings.num_prefix_tokens
            if l >= self.model.num_hidden_layers - self.model.config.num_blocks:
                offset += self.model.config.num_queries
            # offsetting l by +1 becaues hidden_states contains embeddings at the first layer
            token_list.append(outputs["hidden_states"][l + 1][:, offset:])
        tokens = torch.stack(token_list, dim=-1).transpose(1, 2)
        tokens = tokens.reshape(
            tokens.shape[0],
            tokens.shape[1],
            self.model.grid_size[0],
            self.model.grid_size[1],
            tokens.shape[3],
        )
        features = self.merge(tokens)
        features = features.squeeze(-1)

        if return_segmentation:
            # perf-todo: don't scale back to full image size. We only need a downscaled version for DaniFormer
            preds = self.processor.post_process_panoptic_segmentation(
                outputs=outputs, target_sizes=[tuple(img.shape[2:4]) for _ in img]
            )
            return features, preds

        return features


class GenericTimmExtractor(FeatureExtractor):
    def __init__(self, model_name: str, feature_index: int):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(feature_index,),
        )

    def forward(self, img: torch.Tensor):
        # self.model returns a list of feature tensors for the batch
        # in the constructor, out_indices is set, therefore the length of the list is 1
        return self.model(img)[0]
