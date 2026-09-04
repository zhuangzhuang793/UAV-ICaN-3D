"""Low-rank adapters for convolutional Ultralytics YOLO models.

The implementation is deliberately dependency-free: YOLO uses convolutional blocks, while the
popular PEFT defaults primarily target linear/attention projections.  Keeping the adapter here
also makes the merge into a normal Ultralytics checkpoint explicit and testable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAStats:
    adapted_layers: int
    trainable_parameters: int
    total_parameters: int


class LoRAConv2d(nn.Module):
    """Frozen Conv2d plus a trainable low-rank residual branch."""

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if base.groups != 1:
            raise ValueError("grouped convolutions are not supported by LoRAConv2d")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout2d(float(dropout)) if dropout else nn.Identity()
        self.lora_a = nn.Conv2d(
            base.in_channels,
            self.rank,
            base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.lora_b = nn.Conv2d(self.rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_b(self.lora_a(self.dropout(inputs)))
        return self.base(inputs) + self.scaling * residual

    def merged_conv(self) -> nn.Conv2d:
        """Fold the adapter into the base kernel and return a standard Conv2d."""

        delta = torch.einsum(
            "or,rihw->oihw",
            self.lora_b.weight[:, :, 0, 0],
            self.lora_a.weight,
        )
        with torch.no_grad():
            self.base.weight.add_(self.scaling * delta.to(self.base.weight.dtype))
        return self.base


def _set_child(root: nn.Module, qualified_name: str, child: nn.Module) -> None:
    path, _, leaf = qualified_name.rpartition(".")
    parent = root.get_submodule(path) if path else root
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = child
    else:
        setattr(parent, leaf, child)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_pattern: str,
) -> LoRAStats:
    """Replace matching non-grouped convolutions with LoRA residual wrappers."""

    matcher = re.compile(target_pattern)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d) and module.groups == 1 and matcher.search(name)
    ]
    if not targets:
        raise ValueError(f"LoRA target pattern matched no Conv2d layers: {target_pattern!r}")
    for name, module in targets:
        _set_child(model, name, LoRAConv2d(module, rank, alpha, dropout))
    freeze_except_lora(model)
    return lora_stats(model)


def freeze_except_lora(model: nn.Module) -> None:
    """Freeze the base model and leave only adapter matrices trainable."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            module.lora_a.weight.requires_grad = True
            module.lora_b.weight.requires_grad = True


def freeze_batch_norm_statistics(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def merge_lora(model: nn.Module) -> int:
    """Recursively merge all adapters, restoring a regular YOLO module tree."""

    targets = [
        (name, module) for name, module in model.named_modules() if isinstance(module, LoRAConv2d)
    ]
    for name, module in reversed(targets):
        _set_child(model, name, module.merged_conv())
    return len(targets)


def lora_stats(model: nn.Module) -> LoRAStats:
    adapters = [module for module in model.modules() if isinstance(module, LoRAConv2d)]
    return LoRAStats(
        adapted_layers=len(adapters),
        trainable_parameters=sum(
            module.lora_a.weight.numel() + module.lora_b.weight.numel() for module in adapters
        ),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )
