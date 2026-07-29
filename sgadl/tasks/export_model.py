from argparse import ArgumentParser
from pathlib import Path
import re
import torch
import torch.nn as nn
from sgadl import Config, from_config
from sgadl.data.augmentation import get_final_img_size


class Wrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        # ONNX opset 18 does not support antialiasing, so we just disable it for the backbone
        self.base.extractor.model.interpolate_antialias = False

        # no node classification
        self.base.final_node = None

        # obfuscate
        for name, module in self.base.named_modules():
            if name:
                module.__class__.__name__ = "Layer"

    def forward(self, img, pair_ids, box_categories, segmentation, bboxes):
        _, _, rel, p = self.base(
            {
                "img": img,
                "pair_ids": pair_ids,
                "inst2img": torch.zeros((len(box_categories),), dtype=torch.long),
                "box_categories": box_categories,
                "segmentation": segmentation,
                "bboxes": bboxes,
            }
        )
        return rel.sigmoid(), p


def cli():
    parser = ArgumentParser()
    parser.add_argument(
        "model_folder",
        help="The model folder that contains config.json and best_state.pth",
    )
    parser.add_argument("onnx", help="The ONNX output file")
    parser.add_argument(
        "--ema",
        default=False,
        action="store_true",
        help="Use the weights from the EMA model for the ONNX output.",
    )
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    config = Config.from_file(model_folder / "config.json")
    checkpoint = torch.load(model_folder / "best_state.pth", map_location="cpu")

    # determine number of node outputs and number of relation outputs from checkpoint
    final_layers_idx = -1
    num_rel_outputs = -1
    for key, param in checkpoint["model"].items():
        m = re.match(r"final_layers.(\d+).weight", key)
        if m:
            layer_idx = int(m[1])
            if layer_idx > final_layers_idx:
                final_layers_idx = layer_idx
                num_rel_outputs = param.shape[0]
    # divide by 2 because sbj/obj and returned together
    num_node_outputs = checkpoint["model"]["final_node.weight"].shape[0] // 2

    model = from_config.get_model(
        config, num_node_outputs=num_node_outputs, num_rel_outputs=num_rel_outputs
    )
    with torch.no_grad():
        if args.ema:
            model.load_state_dict(checkpoint["ema_model"])
        else:
            model.load_state_dict(checkpoint["model"])

    wrapper = Wrapper(model)

    max_boxes = 20
    max_pairs = max_boxes * (max_boxes - 1)
    h, w = get_final_img_size(from_config.get_augmentations(config, "test"))
    assert w == h
    img_size = h

    # prepare dummy input
    # TODO: don't hardcode, you'll have to change it for the ResNet50 models
    seg = torch.zeros(max_boxes, img_size, img_size, dtype=torch.bool)
    seg[0, 10:20, 10:20] = True
    seg[0, 30:40, 30:40] = True
    bboxes = torch.tensor([(10, 10, 20, 20), (30, 30, 40, 40)], dtype=torch.float32)
    bboxes = torch.cat([bboxes for _ in range(max_boxes // 2)])
    pair_ids = torch.zeros((max_pairs, 2), dtype=torch.long)
    pair_ids[:, 1] = 1
    inputs = {
        "img": torch.rand(1, 3, img_size, img_size),
        "pair_ids": pair_ids,
        "box_categories": torch.zeros((max_boxes,), dtype=torch.long),
        "segmentation": seg,
        "bboxes": bboxes,
    }

    # export to ONNX
    # onnx_program = torch.onnx.dynamo_export(model, inputs)
    onnx_program = torch.onnx.export(
        wrapper,
        (
            inputs["img"],
            inputs["pair_ids"],
            inputs["box_categories"],
            inputs["segmentation"],
            inputs["bboxes"],
        ),
        input_names=[
            "img",
            "pair_ids",
            "box_categories",
            "segmentation",
            "bboxes",
        ],
        dynamo=True,
    )
    onnx_program.save(args.onnx)


if __name__ == "__main__":
    cli()
