# custom building blocks for my networks
from typing import Tuple
import torch
import torch.nn as nn


def norm_coords(coords: torch.Tensor, height_width):
    """Normalises coords to [-1, +1]"""
    h, w = height_width
    coords = coords / torch.tensor((w, h, w, h), device=coords.device)
    coords = 2 * coords - 1
    return coords


class CoordEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = embed_dim // 2
        super().__init__()
        self.coord_embed = nn.Sequential(
            # 8 coordinate values (sbj+obj)
            nn.Linear(in_features=8, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim // 2, out_features=embed_dim),
        )

    def forward(
        self,
        sbj_coords: torch.Tensor,
        obj_coords: torch.Tensor,
        height_width: Tuple[int, int],
    ):
        sbj_coords = norm_coords(sbj_coords, height_width)
        obj_coords = norm_coords(obj_coords, height_width)
        pair_coords = torch.cat((sbj_coords, obj_coords), dim=1)
        return self.coord_embed(pair_coords)
