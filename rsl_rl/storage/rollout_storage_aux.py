# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RolloutStorage extension for auxiliary vision distillation."""

from __future__ import annotations

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage


class RolloutStorageAux(RolloutStorage):
    """Extends RolloutStorage with vision GT targets and vision inputs."""

    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.vision_gt_targets = None
            self.vision_input = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_gt_targets = None
        self.vision_inputs = None

    def add_transitions(self, transition):
        super().add_transitions(transition)
        idx = self.step - 1

        if self.training_type != "distillation":
            return

        if transition.vision_gt_targets is not None:
            vision_gt = transition.vision_gt_targets.detach().to("cpu")
            if self.vision_gt_targets is None:
                vgt_shape = vision_gt.shape[1:]
                self.vision_gt_targets = torch.zeros(
                    self.num_transitions_per_env, self.num_envs, *vgt_shape, dtype=vision_gt.dtype, device="cpu"
                )
            self.vision_gt_targets[idx].copy_(vision_gt)

        if transition.vision_input is not None:
            vision_in = transition.vision_input.detach().to("cpu")
            if self.vision_inputs is None:
                vin_shape = vision_in.shape[1:]
                self.vision_inputs = torch.zeros(
                    self.num_transitions_per_env, self.num_envs, *vin_shape, dtype=vision_in.dtype, device="cpu"
                )
            self.vision_inputs[idx].copy_(vision_in)

    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]

            vision_gt = self.vision_gt_targets[i] if self.vision_gt_targets is not None else None
            vision_in = self.vision_inputs[i] if self.vision_inputs is not None else None

            yield (
                self.observations[i],
                privileged_observations,
                self.actions[i],
                self.privileged_actions[i],
                self.dones[i],
                vision_gt,
                vision_in,
            )
