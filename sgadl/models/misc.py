import torch


def box_union(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Returns the smallest possible box that contains both boxes.
    Box format: ``(x1, y1, x2, y2)``. Box shape: `(B, 4)`
    """
    assert boxes1.shape == boxes2.shape, (boxes1.shape, boxes2.shape)
    assert boxes1.size(1) == 4, boxes1.size(1)
    lt = torch.min(boxes1[..., :2], boxes2[..., :2])
    rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    output = torch.cat((lt, rb), dim=-1)
    return output
