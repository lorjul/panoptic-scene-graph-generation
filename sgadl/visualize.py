import matplotlib.pyplot as plt
from typing import Literal, Optional, Union
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes, draw_keypoints
from torchvision.transforms.functional import pil_to_tensor
from sgadl.data.augmentation import denormalize_imagenet


def draw_pairs(
    img: Union[Image.Image, torch.Tensor],
    bboxes,
    relations=None,
    draw_boxes: Optional[Literal["centre", "box"]] = "box",
    draw_relations=True,
    denorm=False,
    box_width=5,
    centre_radius=5,
    conn_width=5,
    hide_unrelated_boxes=False,
    draw_box_kwargs=None,
):
    assert draw_boxes in (None, "box", "centre"), draw_boxes

    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
    elif isinstance(img, torch.Tensor):
        if img.dtype == torch.float32:
            if denorm:
                img = denormalize_imagenet(img)
            assert img.min() >= 0 and img.max() <= 1, (img.min(), img.max())
            img = (img * 255).to(torch.uint8)
        assert img.dtype == torch.uint8
    else:
        raise RuntimeError("Unsupported img data type")

    if hide_unrelated_boxes:
        assert relations is not None
        used_boxes = set()
        for row in relations:
            sbj, obj = row[:2]
            used_boxes.add(int(sbj))
            used_boxes.add(int(obj))
        used_boxes = sorted(used_boxes)
        remap = {b: i for i, b in enumerate(used_boxes)}
        new_bboxes = []
        for i, box in enumerate(bboxes):
            if i in remap:
                new_bboxes.append(box)
        new_relations = []
        for row in relations:
            sbj = remap[int(row[0])]
            obj = remap[int(row[1])]
            new_relations.append((sbj, obj, *row[2:]))
        relations = new_relations
        bboxes = torch.stack(new_bboxes)

    if draw_boxes == "box":
        if draw_box_kwargs is None:
            draw_box_kwargs = {}
        img = draw_bounding_boxes(
            image=img, boxes=bboxes, width=box_width, **draw_box_kwargs
        )

    if relations is not None and draw_relations:
        if draw_boxes == "centre":
            colors = "red"
        else:
            colors = None

        cx = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
        cy = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
        centres = torch.stack((cx, cy), dim=1)[None]
        conn = [x[:2] for x in relations]
        img = draw_keypoints(
            img,
            keypoints=centres,
            connectivity=conn,
            radius=centre_radius,
            colors=colors,
            width=conn_width,
        )

    return img


def plot_pairs(
    *args,
    fig_kwargs=None,
    **kwargs,
):
    """`args` and `kwargs` get passed to `draw_pairs()`. `fig_kwargs` get passed to `plt.figure()`"""
    if fig_kwargs is None:
        fig_kwargs = {}
    img = draw_pairs(*args, **kwargs)
    fig = plt.figure(**fig_kwargs)
    plt.imshow(img.permute(1, 2, 0))
    return fig
