import torch
import torch.nn as nn


class FreqBias(nn.Module):
    def __init__(self, num_node_classes: int, num_rel_outputs: int):
        super().__init__()
        self.num_node_classes = num_node_classes
        self.node_baseline = nn.Embedding(
            num_node_classes * num_node_classes, num_rel_outputs
        )
        # we could initialize the baseline here with counts from the training set

    def forward(self, sbj_labels: torch.Tensor, obj_labels: torch.Tensor):
        assert (0 <= sbj_labels).all(), sbj_labels.min()
        assert (0 <= obj_labels).all(), obj_labels.min()
        assert (sbj_labels < self.num_node_classes).all(), sbj_labels.max()
        assert (obj_labels < self.num_node_classes).all(), obj_labels.max()
        return self.node_baseline(sbj_labels * self.num_node_classes + obj_labels)
