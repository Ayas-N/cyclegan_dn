# models/dense_norm_core.py
# Thin, device-safe adapter around your DN (from dn.py) so we can plug it into CycleGAN.

from __future__ import annotations
import torch
import torch.nn as nn


from .dense_instance_norm import (
    DenseInstanceNorm,
    PrefetchDenseInstanceNorm,
    init_dense_instance_norm as _init_dn,
    use_dense_instance_norm as _use_dn,
    not_use_dense_instance_norm as _not_use_dn,
    init_prefetch_dense_instance_norm as _init_pdn,
)


def _to_long_on_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.to(device=device, dtype=torch.long, non_blocking=True)
    return torch.as_tensor(t, device=device, dtype=torch.long)


class DenseNormCore(nn.Module):
    """
    Core DN block with a simple API:
       forward(x, x_anchor, y_anchor) -> y
    Wraps your dn.py DenseInstanceNorm and guarantees anchors/buffers are on the right device.
    """
    def __init__(self, num_features: int, affine: bool = True, **_):
        super().__init__()
        # NOTE: your dn.py must NOT hard-code .cuda() in __init__.
        self.core = DenseInstanceNorm(num_features, affine=affine)

    def forward(self, x: torch.Tensor, x_anchor, y_anchor) -> torch.Tensor:
        device = x.device
        xa = _to_long_on_device(x_anchor, device)
        ya = _to_long_on_device(y_anchor, device)

        # Ensure any registered buffers (e.g., mean/std tables) live on the same device as x
        for _, buf in self.core.named_buffers(recurse=True):
            if buf.device != device:
                buf.data = buf.data.to(device, non_blocking=True)

        # Call your DN forward exactly as implemented in dn.py
        return self.core(x, y_anchor=ya, x_anchor=xa)


class PrefetchDenseNormCore(nn.Module):
    """
    Optional: if you want to use the prefetch variant from dn.py.
    Signature matches DenseNormCore.
    """
    def __init__(self, num_features: int, affine: bool = True, **_):
        super().__init__()
        self.core = PrefetchDenseInstanceNorm(num_features, affine=affine)

    def forward(self, x: torch.Tensor, x_anchor, y_anchor) -> torch.Tensor:
        device = x.device
        xa = _to_long_on_device(x_anchor, device)
        ya = _to_long_on_device(y_anchor, device)

        for _, buf in self.core.named_buffers(recurse=True):
            if buf.device != device:
                buf.data = buf.data.to(device, non_blocking=True)

        return self.core(x, y_anchor=ya, x_anchor=xa)


# ---- Convenience pass-through helpers (same names as dn.py) -----------------

def init_dense_norm(model: nn.Module, y_anchor_num: int, x_anchor_num: int) -> None:
    """Initialize DN tables (collection mode)."""
    _init_dn(model, y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num)

def use_dense_norm(model: nn.Module, padding: int = 1) -> None:
    """Switch DN into interpolation/use mode (after collection)."""
    _use_dn(model, padding=padding)

def not_use_dense_norm(model: nn.Module) -> None:
    """Switch DN back to behave like plain instance norm."""
    _not_use_dn(model)

def init_prefetch_dense_norm(model: nn.Module, y_anchor_num: int, x_anchor_num: int) -> None:
    """Initialize prefetch-DN tables (if you use PrefetchDenseNormCore)."""
    _init_pdn(model, y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num)
