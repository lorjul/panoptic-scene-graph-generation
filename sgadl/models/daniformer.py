# transformer architecture for node detection
# easier to validate if it actually works
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from .building_blocks import CoordEncoder
from .feature_extractors import FeatureExtractor
from .transformer_blocks import Transformer, make_sine_position_encoding
from .frequency_bias import FreqBias
from ..utils import box_intersection


def bg_ratio_onoff_rest(sbj_ratios, obj_ratios):
    return torch.zeros_like(sbj_ratios)


def bg_ratio_total_rest(sbj_ratios, obj_ratios):
    return torch.max(torch.tensor(0.0), 1 - sbj_ratios - obj_ratios)


class SbjObjBoxEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        img_size: Tuple[int, int],
        bg_ratio_strategy: str,
        bg_token=True,
    ):
        super().__init__()
        assert bg_ratio_strategy in ("sum", "onoff")
        self.patch_size = patch_size
        self.sbj_token = nn.Parameter(torch.rand(embed_dim))
        self.obj_token = nn.Parameter(torch.rand(embed_dim))
        if bg_token:
            self.background_token = nn.Parameter(torch.rand(embed_dim))
        else:
            self.background_token = nn.Parameter(
                torch.zeros(embed_dim), requires_grad=False
            )
        if bg_ratio_strategy == "sum":
            self.get_bg_ratio = bg_ratio_total_rest
        elif bg_ratio_strategy == "onoff":
            self.get_bg_ratio = bg_ratio_onoff_rest
        else:
            raise ValueError()

        self.onoff = bg_ratio_strategy == "onoff"

        self.patch_coords = self.get_patch_boxes(img_size).reshape(-1, 4)

    def get_patch_boxes(self, img_size):
        rows, cols = torch.meshgrid(
            torch.arange(img_size[0] // self.patch_size),
            torch.arange(img_size[1] // self.patch_size),
            indexing="ij",
        )
        return torch.stack(
            (
                cols * self.patch_size,
                rows * self.patch_size,
                (cols + 1) * self.patch_size,
                (rows + 1) * self.patch_size,
            ),
            dim=-1,
        )

    def forward_ratios(self, sbj_boxes: torch.Tensor, obj_boxes: torch.Tensor):
        ratios = box_intersection(
            torch.cat((sbj_boxes, obj_boxes)), self.patch_coords
        ) / (self.patch_size * self.patch_size)
        ratios = ratios.to(self.sbj_token.device)
        sbj_ratios = ratios[: len(sbj_boxes)]
        obj_ratios = ratios[len(sbj_boxes) :]
        bg_ratios = self.get_bg_ratio(sbj_ratios, obj_ratios)
        return bg_ratios, sbj_ratios, obj_ratios

    def forward(self, sbj_boxes: torch.Tensor, obj_boxes: torch.Tensor):
        bg_ratios, sbj_ratios, obj_ratios = self.forward_ratios(sbj_boxes, obj_boxes)
        if self.onoff:
            sbj_ratios[sbj_ratios > 0] = 1
            obj_ratios[obj_ratios > 0] = 1
            bg_ratios[:] = 0.0
        return (
            bg_ratios[..., None] * normalize(self.background_token, dim=0)[None, None]
            + sbj_ratios[..., None] * normalize(self.sbj_token, dim=0)[None, None]
            + obj_ratios[..., None] * normalize(self.obj_token, dim=0)[None, None]
        )


class SbjObjMaskEncoder(nn.Module):
    """Initial development: Vladyslav Kovganko"""

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        bg_ratio_strategy: str,
        bg_token=True,
    ):
        super().__init__()
        assert bg_ratio_strategy in ("sum", "onoff")
        self.patch_size = patch_size
        self.sbj_token = nn.Parameter(torch.rand(embed_dim))
        self.obj_token = nn.Parameter(torch.rand(embed_dim))
        if bg_token:
            self.background_token = nn.Parameter(torch.rand(embed_dim))
        else:
            self.background_token = nn.Parameter(
                torch.zeros(embed_dim), requires_grad=False
            )

        if bg_ratio_strategy == "sum":
            self.get_bg_ratio = bg_ratio_total_rest
        elif bg_ratio_strategy == "onoff":
            self.get_bg_ratio = bg_ratio_onoff_rest
        else:
            raise ValueError()

        self.onoff = bg_ratio_strategy == "onoff"

    def _pool_ratios(self, seg_masks: torch.Tensor, feature_size: Tuple[int, int]):
        scale_factor_y = seg_masks.size(1) // feature_size[0]
        scale_factor_x = seg_masks.size(2) // feature_size[1]

        kernel_size = (
            self.patch_size * scale_factor_y,
            self.patch_size * scale_factor_x,
        )
        # we need unsqueeze/squeeze for ONNX compatibility
        pooled = nn.functional.avg_pool2d(
            seg_masks.unsqueeze(1).float(), kernel_size=kernel_size, stride=kernel_size
        ).squeeze(1)
        # flatten to return list of patch ratios
        # output shape: (num masks, num patches)
        return pooled.flatten(-2)

    def forward_ratios(
        self,
        seg_masks: torch.Tensor,
        sbj_idx: torch.Tensor,
        obj_idx: torch.Tensor,
        feature_size: Tuple[int, int],
    ):
        with torch.profiler.record_function("Pool Ratios"):
            mask_ratios = self._pool_ratios(seg_masks, feature_size)

        with torch.profiler.record_function("Idx Sbj Ratios"):
            sbj_ratios = mask_ratios[sbj_idx]
        with torch.profiler.record_function("Idx Obj Ratios"):
            obj_ratios = mask_ratios[obj_idx]
        with torch.profiler.record_function("Idx Bg Ratios"):
            bg_ratios = self.get_bg_ratio(sbj_ratios, obj_ratios)
        return bg_ratios, sbj_ratios, obj_ratios

    def forward(
        self,
        seg_masks: torch.Tensor,
        sbj_idx: torch.Tensor,
        obj_idx: torch.Tensor,
        feature_size: Tuple[int, int],
    ):
        bg_ratios, sbj_ratios, obj_ratios = self.forward_ratios(
            seg_masks, sbj_idx, obj_idx, feature_size
        )
        if self.onoff:
            sbj_ratios[sbj_ratios > 0] = 1
            obj_ratios[obj_ratios > 0] = 1
            bg_ratios[:] = 0.0
        # specifying eps is required, otherwise ONNX will crash
        # TODO: this will be probably fixed in the future, so try from time to time if this is not required anymore
        # before, we didn't specify eps at all and just used the default value (1e-12 at the time)
        eps = torch.tensor(1e-12, device=self.background_token.device)
        return (
            bg_ratios[..., None]
            * normalize(self.background_token, dim=0, eps=eps)[None, None]
            + sbj_ratios[..., None]
            * normalize(self.sbj_token, dim=0, eps=eps)[None, None]
            + obj_ratios[..., None]
            * normalize(self.obj_token, dim=0, eps=eps)[None, None]
        )


class DaniFormer(nn.Module):
    def __init__(
        self,
        num_node_outputs: int,
        num_rel_outputs: int,
        extractor: FeatureExtractor,
        transformer_depth=6,
        embed_dim=384,
        patch_size=8,
        feature_shape=(256, 128, 128),
        final_hidden_dim_factor=2,
        use_semantics=False,
        use_masks=False,
        bg_ratio_strategy="total",
        encode_coords=False,
    ):
        super().__init__()
        self.extractor = extractor

        self.patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.feature_shape = feature_shape
        self.use_masks = use_masks

        # perf-todo: embedding the patches again is probably useless
        # since I'm using transformer backends, there is already a patch embedding in the beginning
        # so maybe we can reuse that embedding
        self.patch_embed = nn.Conv2d(
            in_channels=feature_shape[0],
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        pos_encoding = make_sine_position_encoding(
            feature_shape[-2:], patch_size=self.patch_size, d_model=self.embed_dim
        )

        self.transformer_blocks = Transformer(
            transformer_depth=transformer_depth,
            dim=self.embed_dim,
            num_heads=8,
            pos_encoding=pos_encoding,
        )

        # NOTE: usually, the hidden layer is increasing by 4
        final_hidden_dim = round(self.embed_dim * final_hidden_dim_factor)
        self.final_layers = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=final_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=final_hidden_dim, out_features=num_rel_outputs),
        )
        # every relation class gets its parameter output
        self.final_param_layers = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=final_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=final_hidden_dim, out_features=num_rel_outputs),
        )

        self.final_node = nn.Linear(
            in_features=self.embed_dim, out_features=num_node_outputs * 2
        )

        if self.use_masks:
            self.sbjobj_encoder = SbjObjMaskEncoder(
                embed_dim=self.embed_dim,
                patch_size=patch_size,
                bg_ratio_strategy=bg_ratio_strategy,
                bg_token=True,
            )
        else:
            self.sbjobj_encoder = SbjObjBoxEncoder(
                self.embed_dim,
                patch_size,
                feature_shape[1:],
                bg_ratio_strategy=bg_ratio_strategy,
                bg_token=True,
            )

        if encode_coords:
            self.coord_embed = CoordEncoder(self.embed_dim)
        else:
            self.coord_embed = None

        self.classification_token = nn.Parameter(torch.rand(self.embed_dim))

        if use_semantics:
            self.freq_bias = FreqBias(
                num_node_classes=num_node_outputs,
                num_rel_outputs=self.embed_dim,
            )
        else:
            self.freq_bias = None

    def _forward_internal(
        self,
        data: dict,
        features: torch.Tensor,
        patches: torch.Tensor,
        img_shape,
        sbj_ids: torch.Tensor,
        obj_ids: torch.Tensor,
    ):
        inst2img = data["inst2img"]
        box_targets = data["box_categories"]

        with torch.profiler.record_function("Get Img"):
            img_ids = inst2img[sbj_ids]

        with torch.profiler.record_function("SbjObj Encoding"):
            if self.use_masks:
                seg = data["segmentation"]
                sbjobj_tokens = self.sbjobj_encoder(
                    seg, sbj_ids, obj_ids, features.shape[-2:]
                )
            else:
                # scale coords to feature size
                h, w = img_shape[-2:]
                fh, fw = features.shape[1:3]
                coords = data["bboxes"]
                coords[:, 0] *= fw / w
                coords[:, 1] *= fh / h
                coords[:, 2] *= fw / w
                coords[:, 3] *= fh / h
                sbjobj_tokens = self.sbjobj_encoder(coords[sbj_ids], coords[obj_ids])

        extra_tokens = self.classification_token[None, None].expand(
            img_ids.size(0), 1, -1
        )

        # add frequency bias token if requested
        if self.freq_bias is not None:
            with torch.profiler.record_function("Frequency Bias"):
                bias = self.freq_bias(box_targets[sbj_ids], box_targets[obj_ids])
                extra_tokens = torch.cat((extra_tokens, bias[:, None]), dim=1)

        # add coord token if requested
        if self.coord_embed is not None:
            with torch.profiler.record_function("Loc Embedding"):
                coords = data["bboxes"]
                coord_token = self.coord_embed(
                    coords[sbj_ids].to(features.device),
                    coords[obj_ids].to(features.device),
                    img_shape[-2:],
                )
                extra_tokens = torch.cat((extra_tokens, coord_token[:, None]), dim=1)

        with torch.profiler.record_function("Make Tokens"):
            patches_per_box = patches[img_ids]
            tokens = torch.cat((extra_tokens, patches_per_box + sbjobj_tokens), dim=1)

        with torch.profiler.record_function("Transformer Blocks"):
            tokens = self.transformer_blocks(tokens)

        final_token = tokens[:, 0]

        with torch.profiler.record_function("Final Layers"):
            output = self.final_layers(final_token)
            param_output = self.final_param_layers(final_token)

            if self.final_node is None:
                # this is for inference only, safe some memory and skip the node classification
                sbj_cls = None
                obj_cls = None
            else:
                node_output = self.final_node(final_token)
                half_len = node_output.size(1) // 2
                sbj_cls = node_output[:, half_len:]
                obj_cls = node_output[:, :half_len]

        return sbj_cls, obj_cls, output, param_output

    def forward(self, data: dict, predict_segmentation=False):
        """
        Forward pass through the model
        """
        img = data["img"]
        pair_ids = data["pair_ids"]
        sbj_ids = pair_ids[:, 0]
        obj_ids = pair_ids[:, 1]

        with torch.profiler.record_function("Backbone"):
            # if requested, infer the segmentation mask instead of using the ground truth
            # note that the extractor's weights should be completely frozen in this mode,
            # because we are not calculating any loss on the predicted segmentation
            if predict_segmentation:
                features, data["segmentation"] = self.extractor(
                    img, return_segmentation=True
                )
            else:
                features = self.extractor(img)
            assert features.shape[1:] == self.feature_shape, features.shape

        with torch.profiler.record_function("Patch Embedding"):
            patches = self.patch_embed(features).flatten(2).transpose(1, 2)

        return self._forward_internal(
            data=data,
            features=features,
            patches=patches,
            img_shape=img.shape,
            sbj_ids=sbj_ids,
            obj_ids=obj_ids,
        )

    def load_state_dict_partial(self, state_dict):
        """Intended for transfer learning.
        Load the state dict as usual but not the final layers.
        """

        state_dict = dict(state_dict)
        # replace final layer weights with the ones from the model

        # final_node
        for k, v in self.final_node.cpu().state_dict().items():
            k2 = f"final_node.{k}"
            assert k2 in state_dict
            state_dict[k2] = v

        # final_layers
        i = len(self.final_layers) - 1
        for k, v in self.final_layers[i].cpu().state_dict().items():
            k2 = f"final_layers.{i}.{k}"
            assert k2 in state_dict
            state_dict[k2] = v

        # final_param_layers
        i = len(self.final_param_layers) - 1
        for k, v in self.final_param_layers[i].cpu().state_dict().items():
            k2 = f"final_param_layers.{i}.{k}"
            assert k2 in state_dict
            state_dict[k2] = v

        # freq_bias
        if self.freq_bias is not None:
            for k, v in self.freq_bias.cpu().state_dict().items():
                k2 = f"freq_bias.{k}"
                assert k2 in state_dict
                state_dict[k2] = v

        # load the modified weights
        return self.load_state_dict(state_dict)
