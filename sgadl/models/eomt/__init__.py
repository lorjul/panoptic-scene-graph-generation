# check the readme file for how to install EoMT in this repository

from typing import Literal, Union
from pathlib import Path
import torch
import requests
import shutil
import hashlib
from sgadl.models.eomt.eomt import EoMT
from sgadl.models.eomt.vit import ViT

HASHES = {
    ("small", 640): "7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d",
    ("base", 640): "2c8f3296d46f410b85ab64edca34792241d9842dc5031b6b00d4f721a663a8ec",
    ("large", 640): "d48c4a6e51bb04dd5ffa8bf720ccb2ef3ad7b13f8c072ad5a8b6028e7d9e9715",
    ("giant", 640): "0dcb8a7dbd93d9d6512fbc4b4c25f98095bcbe2b7f1c5ab676e594af8908f5c8",
    ("large", 1280): "939bbc0de657929ea762f15d22c9c3f6c455f373564052fc4e0161802aae953b",
    ("giant", 1280): "4e1c074ab0befdcd01a876b34f12e8071c105e954f78d30ff6e4487b81018742",
}


def make_eomt(
    variant: Literal["small", "base", "large", "giant"],
    ckpt_path: Union[Path, str, None] = None,
    img_size: int = 640,
):
    """Create an EoMT model and initialise it with model weights if `ckpt_path` is specified.

    You can usually derive the variant and the image size from the config file name.
    For example, eomt_large_1280.yaml refers to `variant="large"`, `img_size=1280`.

    :param variant: Can be either "small", "base", "large", or "giant". Check the coresponding config file.
    :param ckpt_path: An optional path to a checkpoint file, downloaded from the EoMT repository.
    :param img_size: Expected input image size for the backbone. This is usually 640 or 1280.
    """
    if variant in ("small", "base"):
        num_blocks = 3
    elif variant == "large":
        num_blocks = 4
    elif variant == "giant":
        num_blocks = 5
    else:
        raise ValueError(variant)
    model = EoMT(
        encoder=ViT(
            img_size=(img_size, img_size),
            backbone_name=f"vit_{variant}_patch14_reg4_dinov2",
        ),
        num_classes=133,
        num_q=200,
        num_blocks=num_blocks,
        masked_attn_enabled=False,
    )

    if ckpt_path:
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        new_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("network."):
                k = k[len("network.") :]
                new_ckpt[k] = v
        model.load_state_dict(new_ckpt)

    return model


def download_eomt_ckpt(
    variant: Literal["small", "base", "large", "giant"],
    img_size: int,
    save_dir,
    validate=True,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    if variant == "small" or variant == "base":
        suffix = "_2x"
    else:
        suffix = ""
    url = f"https://huggingface.co/tue-mps/coco_panoptic_eomt_{variant}_{img_size}{suffix}/resolve/main/pytorch_model.bin"
    savename = f"coco_panoptic_eomt_{variant}_{img_size}.pth"
    ckpt_path = save_dir / savename

    # only download if file does not exist
    if not ckpt_path.is_file():
        with requests.get(url, stream=True) as r:
            with open(ckpt_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

    # TODO: verify hash
    hash_func = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)

    if validate:
        assert (variant, img_size) in HASHES
        assert hash_func.hexdigest() == HASHES[(variant, img_size)]

    return ckpt_path
