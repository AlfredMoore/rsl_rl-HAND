# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation algorithm with end-to-end auxiliary vision regression loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.modules.student_teacher_aux import StudentTeacherAux
from rsl_rl.storage.rollout_storage_aux import RolloutStorageAux


class DistillationAux(Distillation):
    """Distillation with auxiliary vision supervision on policy-side vision latent."""

    policy: StudentTeacherAux
    requires_vision_inputs: bool = True

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        aux_loss_weight=0.0,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(
            policy,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            loss_type=loss_type,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.aux_loss_weight = aux_loss_weight
        self.transition = RolloutStorageAux.Transition()

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        self.storage = RolloutStorageAux(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, teacher_obs, vision_input=None, vision_gt_targets=None):
        if vision_input is None:
            raise ValueError("DistillationAux.act requires vision_input")

        self.transition.actions = self.policy.act(obs, vision_input).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        self.transition.vision_input = vision_input
        if vision_gt_targets is not None:
            self.transition.vision_gt_targets = vision_gt_targets
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_aux_loss = 0.0
        loss = 0.0
        cnt = 0

        for _epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions, dones, vision_gt, vision_input in self.storage.generator():
                if vision_input is None:
                    raise RuntimeError("Vision input is missing in rollout storage for DistillationAux update.")
                obs = obs.to(self.device, non_blocking=True)
                privileged_actions = privileged_actions.to(self.device, non_blocking=True)
                vision_input = vision_input.to(self.device, non_blocking=True)
                if vision_gt is not None:
                    vision_gt = vision_gt.to(self.device, non_blocking=True)

                actions = self.policy.act_inference(obs, vision_input)
                behavior_loss = self.loss_fn(actions, privileged_actions)

                aux_loss: torch.Tensor = torch.zeros((), device=self.device)
                if self.aux_loss_weight > 0 and vision_gt is not None:
                    predicted_targets = self.policy.predict_vision_targets()
                    aux_loss = self.loss_fn(predicted_targets, vision_gt)
                    mean_aux_loss += aux_loss.item()

                loss = loss + behavior_loss + self.aux_loss_weight * aux_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0.0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        if cnt == 0:
            raise RuntimeError("No distillation aux batches were generated from rollout storage.")
        mean_behavior_loss /= cnt
        mean_aux_loss /= cnt

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {"behavior": mean_behavior_loss, "aux_vision": mean_aux_loss}
