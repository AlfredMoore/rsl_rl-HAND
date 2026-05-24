# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision encoder for masked-depth student policies.

Architecture (hardcoded — mirrors the validated path in
HAND-policy/scripts/sim2real/train_depth_predictor.py, minus FiLM/FK):

    masked depth (B, 1, H, W) float32, env-normalized to [0, 1]
        → ResNet-10 backbone (4 stages, BasicBlock, BN)
        → 1x1 conv reducing to `num_keypoints` channels
        → SpatialSoftmax → expected (x, y) per channel, normalized to [-1, 1]
        → flatten → (B, num_keypoints * 2)
        → 2-layer MLP head → (B, vision_backbone_dim)

The only externally configurable values are:
    * in_channels         (default 1 — single masked depth channel)
    * vision_backbone_dim (output latent dim — usually 64 / 128 / 256)

H and W are resolved at first forward / runner-side auto-detect from the env's
obs["vision"] shape, never baked into the class. SpatialSoftmax's coordinate
grid is generated inside forward from the runtime feature-map shape, which is
JIT-script friendly (torch.linspace + meshgrid are both supported).

The encoder accepts ALREADY-NORMALIZED input — the env is expected to do
masked depth → clip to [near, far] → min-max scale to [0, 1] BEFORE feeding
the obs into the runner. The encoder never re-clips.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hardcoded hyperparameters from train_depth_predictor.py defaults.
# These are intentionally NOT exposed — the validated checkpoint config maps
# directly to these. If you want to tune them, retrain the encoder from
# scratch on the new arch.
# ---------------------------------------------------------------------------
_BASE_CHANNELS: int = 64        # ResNet stage-0 width
_RESNET10_LAYERS: tuple[int, ...] = (1, 1, 1, 1)
_NUM_KEYPOINTS: int = 64        # 1x1 conv output channel count (spatial softmax K)
_MLP_HIDDEN: int = 256          # MLP head hidden width
_SOFTMAX_TEMPERATURE: float = 1.0


# ---------------------------------------------------------------------------
# ResNet building blocks (lifted from depth_feature.py, FiLM-free)
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    """Standard ResNet basic block (Conv-BN-ReLU x2 + residual)."""

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample: nn.Module = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class _ResNet10(nn.Module):
    """ResNet-10 (4 stages × 1 BasicBlock) without FiLM. JIT-friendly."""

    def __init__(self, in_channels: int = 1, base_channels: int = _BASE_CHANNELS) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        in_ch = base_channels
        for stage_idx, num_blocks in enumerate(_RESNET10_LAYERS):
            out_ch = base_channels * (2 ** stage_idx)
            stride = 1 if stage_idx == 0 else 2
            blocks: list[nn.Module] = [_BasicBlock(in_ch, out_ch, stride=stride)]
            for _ in range(num_blocks - 1):
                blocks.append(_BasicBlock(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.out_channels: int = in_ch    # 64 * 2^(N-1) = 512 for N=4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))


# ---------------------------------------------------------------------------
# Spatial softmax (Levine et al. 2016)
# ---------------------------------------------------------------------------

class _SpatialSoftmax(nn.Module):
    """Spatial-softmax → expected (x, y) per channel in [-1, 1].

    Coordinate grid is generated inside forward from runtime H, W so the
    encoder doesn't need to know the input resolution at construction time
    (auto-detect path). torch.linspace + meshgrid are both jit.script-supported.
    """

    def __init__(self, temperature: float = _SOFTMAX_TEMPERATURE) -> None:
        super().__init__()
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        x_flat = x_grid.reshape(1, 1, -1)
        y_flat = y_grid.reshape(1, 1, -1)

        logits = x.reshape(B, C, -1) / self.temperature
        attn = F.softmax(logits, dim=-1)
        kx = (attn * x_flat).sum(dim=-1)
        ky = (attn * y_flat).sum(dim=-1)
        return torch.stack([kx, ky], dim=-1)        # (B, C, 2)


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """ResNet-10 + spatial softmax + MLP head → (B, vision_backbone_dim).

    Hardcoded internals (intentional — no YAML knob):
        backbone        = ResNet-10 (4 stages × 1 BasicBlock)
        base_channels   = 64
        num_keypoints   = 64
        mlp_hidden      = 256
        spatial_softmax temperature = 1.0 (fixed buffer, non-learnable)

    Externally configurable:
        in_channels         (default 1, single masked depth channel)
        vision_backbone_dim (output latent dim)

    Input contract:
        x : (B, in_channels, H, W) float32, env-normalized to [0, 1].
            The encoder does NOT clip or rescale — env owns preprocessing.
            H, W are arbitrary but must be large enough to survive 4× stride
            downsampling (>= ~32px is the practical minimum; 120x120 / 240x320
            are typical).

    Output:
        (B, vision_backbone_dim) float32 latent.
    """

    def __init__(
        self,
        in_channels: int = 1,
        vision_backbone_dim: int = 128,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if vision_backbone_dim < 1:
            raise ValueError(f"vision_backbone_dim must be >= 1, got {vision_backbone_dim}")

        self.in_channels: int = int(in_channels)
        self.vision_backbone_dim: int = int(vision_backbone_dim)

        # ── network ─────────────────────────────────────────────────────
        self.backbone = _ResNet10(in_channels=in_channels, base_channels=_BASE_CHANNELS)
        # 1x1 reduces ResNet's final channel count (512) to num_keypoints.
        self.kp_conv = nn.Conv2d(self.backbone.out_channels, _NUM_KEYPOINTS, kernel_size=1)
        self.spatial_softmax = _SpatialSoftmax(temperature=_SOFTMAX_TEMPERATURE)

        # head input = K keypoints × 2 (xy) → MLP → backbone_dim
        self.head = nn.Sequential(
            nn.Linear(_NUM_KEYPOINTS * 2, _MLP_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(_MLP_HIDDEN, vision_backbone_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, H, W) → (B, vision_backbone_dim)."""
        feat = self.backbone(x)                      # (B, 512, h, w)
        kp_map = self.kp_conv(feat)                  # (B, K, h, w)
        keypoints = self.spatial_softmax(kp_map)     # (B, K, 2)
        kp_flat = keypoints.flatten(1)               # (B, K * 2)
        return self.head(kp_flat)                    # (B, vision_backbone_dim)
