import math
import torch
import torch.nn as nn


def make_sine_position_encoding(
    in_feature_size, patch_size, d_model, temperature=10000, scale=2 * math.pi
):
    # number of patches per height and width
    h, w = in_feature_size[0] // (patch_size[0]), in_feature_size[1] // (patch_size[1])
    area = torch.ones(1, h, w)  # [b, h, w]
    # 1, 2, 3, 4, 5, ... in x-direction
    y_embed = area.cumsum(1, dtype=torch.float32)
    # 1, 2, 3, 4, 5, ... in y-direction
    x_embed = area.cumsum(2, dtype=torch.float32)

    one_direction_feats = d_model // 2

    eps = 1e-6
    # equally spaced entries between 0 and scale in y-direction
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    # equally spaced entries between 0 and scale in x-direction
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
    # temperature ** (half of embed size equally spaced double entries (always to identical entries)
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="floor") / one_direction_feats
    )

    # adds embedding_size // 2 dimension in the end. smooth transition from largest value scale (pos_x[:, :, -1, 0]) to 0 (pos_x[:, :, 0, -1]) with exponential drop
    pos_x = x_embed[:, :, :, None] / dim_t
    # same, but dimension 1 and 2 swapped
    pos_y = y_embed[:, :, :, None] / dim_t
    # sine and cos wave from [:, :, 0, -1] to [:, :, -1, 0] becoming "slower" at [:, :, 0, -1]. Only every second entry taken but there are always two identical (for sine and cos). After stack and flatten, alternating sine and cosine values
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    # combination of sine and cosine waves. in the embedding direction, the first halt (up to 96) is y-direction, then x-direction. the embedding direction is shifted to dimension 1
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    # put the patches together
    pos = pos.flatten(2).permute(0, 2, 1)
    return pos


class Transformer(nn.Module):
    def __init__(
        self,
        transformer_depth,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        act_layer="gelu",
        pos_encoding=None,
    ):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=int(mlp_ratio * dim),
                    dropout=drop,
                    activation=act_layer,
                    batch_first=True,
                    norm_first=True,
                    bias=True,
                )
                for _ in range(transformer_depth)
            ]
        )
        if pos_encoding is None:
            self.pos_encoding = None
        else:
            assert isinstance(pos_encoding, torch.Tensor)
            self.pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)

    def forward(self, tokens):
        for block in self.transformer_blocks:
            if self.pos_encoding is not None:
                tokens[:, -self.pos_encoding.size(1) :] += self.pos_encoding
            tokens = block(tokens)

        return tokens
