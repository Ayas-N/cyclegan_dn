import torch.nn as nn

from .dense_norm_core import DenseNormCore        # your adapter
from . import dense_context

class DenseNorm2d(nn.Module):
    def __init__(self, num_features, **_):
        super().__init__()
        self.core = DenseNormCore(num_features, affine=True)

    def forward(self, x):
        B = x.size(0)
        xa, ya = dense_context.get_anchors(x.device, B)
        return self.core(x, x_anchor=xa, y_anchor=ya)
