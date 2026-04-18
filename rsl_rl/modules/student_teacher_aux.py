# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""StudentTeacher with policy-side vision encoder and auxiliary vision head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.modules.student_teacher import StudentTeacher


def _conv2d_out_hw(h: int, w: int, kernel_size: int, stride: int, padding: int = 0) -> tuple[int, int]:
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    return out_h, out_w


class _VisionEncoder2D(nn.Module):
    """Lightweight 4-layer CNN encoder with global pooling."""

    def __init__(self, in_channels: int, latent_dim: int, input_height: int, input_width: int):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim

        h1, w1 = _conv2d_out_hw(self.input_height, self.input_width, kernel_size=6, stride=2)
        h2, w2 = _conv2d_out_hw(h1, w1, kernel_size=4, stride=2)
        h3, w3 = _conv2d_out_hw(h2, w2, kernel_size=4, stride=2)
        h4, w4 = _conv2d_out_hw(h3, w3, kernel_size=3, stride=2)

        c1, c2, c3, c4 = 16, 32, 64, latent_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([c1, h1, w1]),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([c2, h2, w2]),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([c3, h3, w3]),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([c4, h4, w4]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        return feat.view(-1, self.latent_dim)


class StudentTeacherAux(StudentTeacher):
    """StudentTeacher with e2e vision encoder and auxiliary target head.

    Behavior branch:
        actions = student([proprio_obs, vision_latent])

    Auxiliary branch:
        vision_targets = vision_head(vision_latent)
    """

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        vision_backbone_dim=64,
        vision_target_dim=10,
        student_vision_modality="rgb",
        vision_input_height=120,
        vision_input_width=120,
        depth_clip_min=0.1,
        depth_clip_max=2.0,
        **kwargs,
    ):
        self.num_proprio_obs = int(num_student_obs)
        self.vision_backbone_dim = int(vision_backbone_dim)
        self.student_vision_modality = str(student_vision_modality)
        self.vision_input_height = int(vision_input_height)
        self.vision_input_width = int(vision_input_width)
        self.depth_clip_min = float(depth_clip_min)
        self.depth_clip_max = float(depth_clip_max)

        if self.vision_backbone_dim <= 0:
            raise ValueError(f"vision_backbone_dim must be > 0, got {self.vision_backbone_dim}")
        if vision_target_dim <= 0:
            raise ValueError(f"vision_target_dim must be > 0, got {vision_target_dim}")
        if self.depth_clip_max <= self.depth_clip_min:
            raise ValueError(
                "Invalid depth clip range: "
                f"depth_clip_min={self.depth_clip_min}, depth_clip_max={self.depth_clip_max}"
            )
        if self.student_vision_modality not in {"rgb", "depth"}:
            raise ValueError(f"Invalid student_vision_modality: {self.student_vision_modality}")

        # Student MLP consumes fused [proprio, vision_latent].
        fused_student_obs = self.num_proprio_obs + self.vision_backbone_dim
        super().__init__(
            fused_student_obs,
            num_teacher_obs,
            num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        in_channels = 3 if self.student_vision_modality == "rgb" else 1
        self.vision_encoder = _VisionEncoder2D(
            in_channels=in_channels,
            latent_dim=self.vision_backbone_dim,
            input_height=self.vision_input_height,
            input_width=self.vision_input_width,
        )
        self.vision_head = nn.Linear(self.vision_backbone_dim, vision_target_dim)
        self._last_vision_latent = None

        # RGB normalization (ImageNet stats)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

        print(
            f"[StudentTeacherAux] Modality={self.student_vision_modality}, "
            f"vision latent={self.vision_backbone_dim}, aux target={vision_target_dim}"
        )

    def _prepare_rgb(self, vision_input: torch.Tensor) -> torch.Tensor:
        if vision_input.ndim != 4:
            raise ValueError(f"RGB input must be 4D, got shape={tuple(vision_input.shape)}")

        # Accept (N,H,W,C) from env.
        if vision_input.shape[-1] >= 3:
            x = vision_input[..., :3].permute(0, 3, 1, 2).contiguous()
        elif vision_input.shape[1] == 3:
            x = vision_input
        else:
            raise ValueError(f"Unsupported RGB input shape: {tuple(vision_input.shape)}")

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if x.max() > 1.0:
                x = x / 255.0

        if x.shape[-2] != self.vision_input_height or x.shape[-1] != self.vision_input_width:
            x = F.interpolate(
                x,
                size=(self.vision_input_height, self.vision_input_width),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - self.rgb_mean) / self.rgb_std
        return x

    def _prepare_depth(self, vision_input: torch.Tensor) -> torch.Tensor:
        if vision_input.ndim == 3:
            x = vision_input.unsqueeze(1)
        elif vision_input.ndim == 4 and vision_input.shape[-1] == 1:
            x = vision_input.permute(0, 3, 1, 2).contiguous()
        elif vision_input.ndim == 4 and vision_input.shape[1] == 1:
            x = vision_input
        else:
            raise ValueError(f"Unsupported depth input shape: {tuple(vision_input.shape)}")

        # uint8 path corresponds to [clip_min, clip_max] quantization.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
            x = self.depth_clip_min + x * (self.depth_clip_max - self.depth_clip_min)
        else:
            x = x.float()

        if x.shape[-2] != self.vision_input_height or x.shape[-1] != self.vision_input_width:
            x = F.interpolate(
                x,
                size=(self.vision_input_height, self.vision_input_width),
                mode="bilinear",
                align_corners=False,
            )
        x = x.clamp_(min=self.depth_clip_min, max=self.depth_clip_max)
        x = (x - self.depth_clip_min) / (self.depth_clip_max - self.depth_clip_min)
        return x

    def _encode_vision(self, vision_input: torch.Tensor) -> torch.Tensor:
        if vision_input is None:
            raise ValueError("vision_input is required for StudentTeacherAux")
        if self.student_vision_modality == "rgb":
            x = self._prepare_rgb(vision_input)
        else:
            x = self._prepare_depth(vision_input)
        return self.vision_encoder(x)

    def _build_student_input(self, observations: torch.Tensor, vision_input: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.num_proprio_obs:
            raise ValueError(
                "Unexpected proprio observation dimension. "
                f"got={observations.shape[-1]}, expected={self.num_proprio_obs}"
            )
        vision_latent = self._encode_vision(vision_input)
        self._last_vision_latent = vision_latent
        return torch.cat((observations, vision_latent), dim=-1)

    def act_inference(self, observations: torch.Tensor, vision_input: torch.Tensor):
        student_in = self._build_student_input(observations, vision_input)
        return self.student(student_in)

    def act(self, observations: torch.Tensor, vision_input: torch.Tensor):
        student_in = self._build_student_input(observations, vision_input)
        mean = self.student(student_in)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def predict_vision_targets(self, vision_latent: torch.Tensor | None = None) -> torch.Tensor:
        if vision_latent is None:
            vision_latent = self._last_vision_latent
        if vision_latent is None:
            raise RuntimeError("Vision latent not available. Call act()/act_inference() first.")
        return self.vision_head(vision_latent)

    def get_last_vision_latent(self) -> torch.Tensor | None:
        return self._last_vision_latent

    def load_vision_head_from_cnn(
        self,
        cnn_state_dict: dict[str, torch.Tensor],
        target_indices: Sequence[int] | None = None,
    ):
        """Optional compatibility loader from a phase-1 CNN linear head."""
        weight = cnn_state_dict["linear.0.weight"]
        bias = cnn_state_dict["linear.0.bias"]

        if target_indices is not None:
            index = torch.as_tensor(target_indices, dtype=torch.long, device=weight.device)
            weight = weight.index_select(0, index)
            bias = bias.index_select(0, index)

        if tuple(weight.shape) != tuple(self.vision_head.weight.shape):
            raise ValueError(
                "CNN head shape mismatch. "
                f"expected={tuple(self.vision_head.weight.shape)}, got={tuple(weight.shape)}"
            )
        if tuple(bias.shape) != tuple(self.vision_head.bias.shape):
            raise ValueError(
                "CNN head bias shape mismatch. "
                f"expected={tuple(self.vision_head.bias.shape)}, got={tuple(bias.shape)}"
            )
        self.vision_head.load_state_dict({"weight": weight, "bias": bias})
        print("[INFO]: Loaded vision head weights from CNN checkpoint")

    def load_vision_encoder_from_cnn(self, cnn_state_dict: dict[str, torch.Tensor]):
        """Load vision encoder backbone weights from a Phase-1 CNN checkpoint.

        The Phase-1 checkpoint (AuxPhase1CNN) stores encoder weights under
        ``encoder.*`` keys. This method strips the prefix and loads them into
        ``self.vision_encoder``.
        """
        prefix = "encoder."
        encoder_sd = {
            k.removeprefix(prefix): v
            for k, v in cnn_state_dict.items()
            if k.startswith(prefix)
        }
        if not encoder_sd:
            raise ValueError(
                "No encoder.* keys found in checkpoint. "
                f"Available keys: {sorted(cnn_state_dict.keys())[:10]}"
            )
        self.vision_encoder.load_state_dict(encoder_sd)
        print(f"[INFO]: Loaded vision encoder weights from CNN checkpoint ({len(encoder_sd)} tensors)")

    def load_from_phase1_cnn(
        self,
        cnn_state_dict: dict[str, torch.Tensor],
        target_indices: Sequence[int] | None = None,
    ):
        """Convenience method: load both encoder and head from Phase-1 checkpoint."""
        self.load_vision_encoder_from_cnn(cnn_state_dict)
        self.load_vision_head_from_cnn(cnn_state_dict, target_indices=target_indices)
