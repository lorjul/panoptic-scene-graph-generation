from typing import Union, Sequence
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def denormalize_imagenet(img: torch.Tensor):
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    return img * std + mean


def get_final_img_size(tfm):
    """Returns image size as `(height, width)`."""
    # assuming that the augmentation pipeline always produces the same image size
    dummy = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    transformed = tfm(dummy)
    return tuple(transformed.shape[1:].tolist())


class SquarePad(T.Transform):
    def _get_params(self, flat_inputs):
        # for compatibility with torchvision < 0.21
        return self.make_params(flat_inputs)

    def _transform(self, inpt, params):
        # for compatibility with torchvision < 0.21
        return self.transform(inpt, params)

    def make_params(self, flat_inputs):
        h, w = T.query_size(flat_inputs)
        size = max(h, w)
        # padding on: left, top, right, bottom
        return {"padding": [0, 0, size - w, size - h]}

    def transform(self, inpt, params):
        if isinstance(inpt, (torch.Tensor, tv_tensors.Mask, tv_tensors.Image)):
            return F.pad(inpt, params["padding"])
        elif isinstance(inpt, tv_tensors.BoundingBoxes):
            return inpt
        raise NotImplementedError("Unsupported tensor type")


class BoxJitter(T.Transform):
    def __init__(self, max_relative_offset: float = 0.1):
        super().__init__()
        assert 0 <= max_relative_offset < 1
        self.max_relative_offset = max_relative_offset

    def _transform(self, inpt, params):
        # for compatibility with torchvision < 0.21
        return self.transform(inpt, params)

    def transform(self, inpt, params):
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            assert inpt.format == tv_tensors.BoundingBoxFormat.XYXY, inpt.format

            offsets = (2 * torch.rand(inpt.shape) - 1) * self.max_relative_offset
            widths_heights = (inpt - inpt[:, (2, 3, 0, 1)]).abs()
            return inpt + offsets * widths_heights

        return inpt


def BoxClamp():
    """Alias for ClampBoundingBoxes"""
    return T.ClampBoundingBoxes()


def Prob(tfm, prob: float = 0.5):
    """Alias for RandomApply"""
    return T.RandomApply([tfm], p=prob)


def ToTensor():
    """Alias for ToDtype(dtype=torch.float32, scale=True).
    After this step, the image will be a PyTorch tensor with coefficients in [0, 1].
    """
    return T.ToDtype(dtype=torch.float32, scale=True)


def NormalizeFor(stats: str):
    """Alias for T.Normalize"""
    if stats == "clip":
        return T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
    elif stats == "imagenet":
        return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    raise ValueError(f"Unknown stats name: {stats}")


def config_to_aug(config: Union[dict, str]):
    TRANSFORMS = [
        SquarePad,
        T.Resize,
        T.ToDtype,
        T.RandomHorizontalFlip,
        BoxJitter,
        T.ClampBoundingBoxes,
        T.Identity,
        T.RandomApply,
        T.Normalize,
        T.ColorJitter,
        T.RandomSolarize,
        T.RandomGrayscale,
        T.GaussianBlur,
        T.RandomChoice,
        # alias
        ToTensor,
        BoxClamp,
        Prob,
        NormalizeFor,
    ]

    if isinstance(config, str):
        for cls in TRANSFORMS:
            if cls.__name__ == config:
                return cls()
        raise RuntimeError(f"Unknown transform: {config}")

    # parse augmentations with parameters
    for tfm_cls in TRANSFORMS:
        if tfm_cls.__name__ == config["@"]:
            break
    else:
        raise RuntimeError("Unknown transform name")

    kwargs = {}
    for k, v in config.items():
        if k == "@":
            continue
        # it's a nested augmentation!
        if isinstance(v, dict) and "@" in v:
            v = config_to_aug(v)
        elif isinstance(v, Sequence):
            new_v = []
            contains_notfm = False
            for x in v:
                if isinstance(x, dict) and "@" in x:
                    new_v.append(config_to_aug(x))
                else:
                    contains_notfm = True
            if len(new_v) > 0:
                if contains_notfm:
                    raise RuntimeError(
                        "Augmentation list contains a mix of nested transforms and other items"
                    )
                else:
                    # overwrite sequence with list of nested augmentations
                    v = new_v
            # if new_v is empty, the sequence will be used as-is (e.g. as parameters for a transform)
        kwargs[k] = v
    return tfm_cls(**kwargs)


def get_standard_transforms(is_train: bool, image_size: int, box_jitter=False):
    jitter = T.Identity()
    if box_jitter and is_train:
        jitter = T.RandomApply([BoxJitter()], p=0.7)

    return [
        T.ToDtype(dtype=torch.float32, scale=True),
        jitter,
        T.ClampBoundingBoxes(),
        SquarePad(),
        T.Resize(image_size),
        (
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
            if is_train
            else T.Identity()
        ),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
