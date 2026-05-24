# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.vision_encoder import VisionEncoder


class StudentTeacher(nn.Module):
    """MLP student + MLP teacher, with optional vision-encoder branch on the student.

    Backward-compatible: if ``vision_backbone_dim`` is left at its default (None),
    the class behaves exactly like the original MLP-only student/teacher and all
    older callers (``act(obs)`` / ``act_inference(obs)``) keep working.

    With vision enabled, the student's MLP input is
        cat([proprio_obs, vision_encoder(vision)], dim=-1)
    and the runner is expected to pass ``vision`` through ``act`` /
    ``act_inference`` as a second argument. The teacher never sees vision; it
    still consumes 1D ``teacher_observations`` only.
    """

    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        # ── Vision branch (optional; None disables vision entirely) ──────
        vision_backbone_dim: int | None = None,
        vision_in_channels: int = 1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        # ── Vision encoder (optional) ────────────────────────────────────
        # When None: pure MLP student, behaves identically to legacy code.
        # When int:  ResNet-10 + spatial-softmax encoder; student MLP input
        #            = num_student_obs (proprio) + vision_backbone_dim.
        if vision_backbone_dim is not None:
            self.has_vision: bool = True
            self.vision_backbone_dim: int = int(vision_backbone_dim)
            self.vision_encoder = VisionEncoder(
                in_channels=vision_in_channels,
                vision_backbone_dim=int(vision_backbone_dim),
            )
            student_input_dim = int(num_student_obs) + int(vision_backbone_dim)
        else:
            self.has_vision = False
            self.vision_backbone_dim = 0
            self.vision_encoder = None
            student_input_dim = int(num_student_obs)

        mlp_input_dim_s = student_input_dim
        mlp_input_dim_t = num_teacher_obs

        # student
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        # teacher (always pure MLP — teacher never sees vision)
        teacher_layers = []
        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")
        if self.has_vision:
            print(f"Vision encoder: ResNet10+spatial, latent_dim={self.vision_backbone_dim}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # ------------------------------------------------------------------
    # Student input fusion (proprio [+ vision latent])
    # ------------------------------------------------------------------
    def _build_student_input(
        self,
        observations: torch.Tensor,
        vision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse 1D proprio + (optional) vision latent into the student MLP input.

        - has_vision=False: returns observations as-is. ``vision`` is ignored
          if accidentally passed (warns once would be noisy — silently ignore).
        - has_vision=True:  vision must be a (B, C, H, W) tensor; encoder runs
          and the latent is concatenated to observations along the last dim.
        """
        if not self.has_vision:
            return observations
        if vision is None:
            raise ValueError(
                "StudentTeacher.has_vision=True but vision input was not provided. "
                "Pass vision=<Tensor> to act / act_inference."
            )
        vision_latent = self.vision_encoder(vision)
        return torch.cat([observations, vision_latent], dim=-1)

    def update_distribution(
        self,
        observations: torch.Tensor,
        vision: torch.Tensor | None = None,
    ) -> None:
        student_in = self._build_student_input(observations, vision)
        mean = self.student(student_in)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(
        self,
        observations: torch.Tensor,
        vision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.update_distribution(observations, vision=vision)
        return self.distribution.sample()

    def act_inference(
        self,
        observations: torch.Tensor,
        vision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        student_in = self._build_student_input(observations, vision)
        return self.student(student_in)

    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        # Teacher never sees vision. 1D privileged obs in, action out.
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load student/teacher parameters with the same dual semantics as before.

        Two accepted formats:
          1. PPO-trained actor-critic checkpoint (keys contain ``actor.*``):
             load only the actor weights into ``self.teacher``. The vision
             encoder / student weights are left at their init values — this is
             the standard "drop in a PPO teacher, distill a vision student"
             flow. Returns False (not a resume).
          2. Distillation checkpoint (keys contain ``student.*``):
             load via super().load_state_dict — all student/teacher (+ vision
             encoder if present in the file) weights are loaded together.
             Returns True (resume).
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    # ------------------------------------------------------------------
    # JIT export — returns a TorchScript-friendly wrapper depending on
    # whether the policy has a vision branch.
    # ------------------------------------------------------------------
    def as_jit(self, normalizer: nn.Module | None = None) -> nn.Module:
        """Return a TorchScript-friendly wrapper of the student-side inference path.

        Vision policy:
            forward(proprio, vision) -> action_mean
        MLP-only policy:
            forward(observations) -> action_mean

        Deployment-side usage (no rsl_rl dependency):
            policy = torch.jit.load("policy.pt")
            action = policy(proprio, vision)        # if has_vision
            action = policy(observations)            # otherwise
        """
        if self.has_vision:
            return _StudentVisionJitWrapper(self, normalizer)
        return _StudentMLPJitWrapper(self, normalizer)


# ---------------------------------------------------------------------------
# JIT wrappers (private)
# ---------------------------------------------------------------------------

class _StudentMLPJitWrapper(nn.Module):
    """Bundles obs_normalizer + student MLP for a non-vision policy.

    forward(observations) -> action_mean.
    Equivalent to IsaacLab-HAND's legacy _TorchPolicyExporter (non-recurrent).

    Wrapper is forced into eval mode at construction so EmpiricalNormalization
    skips its training-only `update` branch (which is @torch.jit.unused and
    crashes the scripted module if reached).
    """

    def __init__(self, policy: StudentTeacher, normalizer: nn.Module | None = None) -> None:
        super().__init__()
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()
        self.student = copy.deepcopy(policy.student)
        self.eval()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.student(self.normalizer(observations))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _StudentVisionJitWrapper(nn.Module):
    """Bundles obs_normalizer + vision encoder + student MLP for a vision policy.

    forward(proprio, vision) -> action_mean.
    vision input is expected to be ALREADY env-normalized (mask + clip + scale
    to [0, 1]); the wrapper does NOT re-clip / re-normalize.

    Same eval-mode-at-construction pattern as the MLP wrapper.
    """

    def __init__(self, policy: StudentTeacher, normalizer: nn.Module | None = None) -> None:
        super().__init__()
        if not policy.has_vision:
            raise ValueError("_StudentVisionJitWrapper requires a vision-enabled StudentTeacher.")
        self.proprio_normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()
        self.vision_encoder = copy.deepcopy(policy.vision_encoder)
        self.student = copy.deepcopy(policy.student)
        self.eval()

    def forward(self, proprio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        proprio_norm = self.proprio_normalizer(proprio)
        vision_latent = self.vision_encoder(vision)
        return self.student(torch.cat([proprio_norm, vision_latent], dim=-1))

    @torch.jit.export
    def reset(self) -> None:
        pass
