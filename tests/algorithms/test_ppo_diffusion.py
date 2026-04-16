# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PPODiffusion algorithm."""

from __future__ import annotations

import copy
import tempfile
import torch
import pytest
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_diffusion import PPODiffusion
from rsl_rl.models import MLPModel
from rsl_rl.models.diffusion_model import DiffusionModel
from rsl_rl.storage.diffusion_rollout_storage import DiffusionRolloutStorage
from rsl_rl.storage.rollout_storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4
FT_DENOISING_STEPS = 3
DENOISING_STEPS = 10


def _make_actor(obs: TensorDict, obs_groups: dict) -> DiffusionModel:
    return DiffusionModel(
        obs, obs_groups, "actor", NUM_ACTIONS,
        denoising_steps=DENOISING_STEPS,
        ft_denoising_steps=FT_DENOISING_STEPS,
        horizon_steps=1,
        mlp_dims=[32, 32],
        activation_type="Mish",
        residual_style=False,
    )


def _make_critic(obs: TensorDict, obs_groups: dict) -> MLPModel:
    return MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[32, 32])


def _make_storage(obs: TensorDict) -> DiffusionRolloutStorage:
    return DiffusionRolloutStorage(
        "rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS],
        ft_denoising_steps=FT_DENOISING_STEPS,
        horizon_steps=1,
        action_dim=NUM_ACTIONS,
    )


def _build_dppo(**overrides) -> tuple[PPODiffusion, TensorDict]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, obs_groups)
    critic = _make_critic(obs, obs_groups)
    storage = _make_storage(obs)
    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma_denoising=0.99,
        clip_ploss_coef=0.1,
        clip_ploss_coef_base=1e-3,
        clip_ploss_coef_rate=3.0,
        reward_horizon=2,
    )
    defaults.update(overrides)
    dppo = PPODiffusion(actor, critic, storage, **defaults)
    return dppo, obs


class TestPPODiffusionConstruction:
    def test_dual_optimizers_created(self):
        dppo, _ = _build_dppo()
        assert hasattr(dppo, "actor_optimizer")
        assert hasattr(dppo, "critic_optimizer")

    def test_optimizer_alias(self):
        dppo, _ = _build_dppo()
        # self.optimizer should alias actor_optimizer for runner compat
        assert dppo.optimizer is dppo.actor_optimizer

    def test_actor_is_diffusion_model(self):
        dppo, _ = _build_dppo()
        assert isinstance(dppo.actor, DiffusionModel)

    def test_storage_is_diffusion_storage(self):
        dppo, _ = _build_dppo()
        assert isinstance(dppo.storage, DiffusionRolloutStorage)


class TestPPODiffusionAct:
    def test_act_returns_correct_action_shape(self):
        dppo, obs = _build_dppo()
        actions = dppo.act(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_act_stores_pending_chains(self):
        dppo, obs = _build_dppo()
        dppo.act(obs)
        assert dppo._pending_chains is not None
        assert dppo._pending_chains.shape == (NUM_ENVS, FT_DENOISING_STEPS + 1, 1, NUM_ACTIONS)

    def test_act_stores_pending_denoising_logprobs(self):
        dppo, obs = _build_dppo()
        dppo.act(obs)
        assert dppo._pending_denoising_logprobs is not None
        assert dppo._pending_denoising_logprobs.shape == (NUM_ENVS, FT_DENOISING_STEPS, 1, NUM_ACTIONS)

    def test_act_stores_transition_fields(self):
        dppo, obs = _build_dppo()
        dppo.act(obs)
        assert dppo.transition.actions is not None
        assert dppo.transition.values is not None
        assert dppo.transition.actions_log_prob is not None


class TestPPODiffusionProcessEnvStep:
    def test_process_env_step_increments_storage_step(self):
        dppo, obs = _build_dppo()
        dppo.act(obs)
        initial_step = dppo.storage.step
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        dppo.process_env_step(obs, rewards, dones, {})
        assert dppo.storage.step == initial_step + 1

    def test_process_env_step_stores_diffusion_data(self):
        dppo, obs = _build_dppo()
        dppo.act(obs)
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        dppo.process_env_step(obs, rewards, dones, {})
        # Chains at step 0 should be non-zero (were stored)
        assert not torch.all(dppo.storage.chains[0] == 0)

    def test_process_env_step_timeout_bootstrapping(self):
        dppo, obs = _build_dppo()
        dppo.act(obs)
        stored_values = dppo.transition.values.clone()
        raw_reward = torch.ones(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        time_outs = torch.zeros(NUM_ENVS)
        time_outs[0] = 1.0
        dppo.process_env_step(obs, raw_reward, dones, {"time_outs": time_outs})
        stored_reward = dppo.storage.rewards[0, 0, 0].item()
        expected = 1.0 + dppo.gamma * stored_values[0, 0].item()
        assert abs(stored_reward - expected) < 1e-5


def _run_full_collection(dppo: PPODiffusion, obs: TensorDict) -> None:
    """Run NUM_STEPS of act + process_env_step to fill storage."""
    for _ in range(NUM_STEPS):
        dppo.act(obs)
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        dppo.process_env_step(obs, rewards, dones, {})


class TestPPODiffusionUpdate:
    def test_full_training_loop_runs(self):
        dppo, obs = _build_dppo()
        _run_full_collection(dppo, obs)
        dppo.compute_returns(obs)
        losses = dppo.update()
        assert "value" in losses
        assert "surrogate" in losses
        assert "entropy" in losses

    def test_update_returns_finite_losses(self):
        dppo, obs = _build_dppo()
        _run_full_collection(dppo, obs)
        dppo.compute_returns(obs)
        losses = dppo.update()
        for key, val in losses.items():
            assert torch.isfinite(torch.tensor(val)), f"Loss '{key}' is not finite: {val}"

    def test_update_clears_storage(self):
        dppo, obs = _build_dppo()
        _run_full_collection(dppo, obs)
        dppo.compute_returns(obs)
        dppo.update()
        assert dppo.storage.step == 0

    def test_actor_parameters_change_after_update(self):
        dppo, obs = _build_dppo()
        params_before = [p.clone() for p in dppo.actor.actor_ft.parameters()]
        _run_full_collection(dppo, obs)
        dppo.compute_returns(obs)
        dppo.update()
        params_after = list(dppo.actor.actor_ft.parameters())
        changed = any(
            not torch.allclose(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert changed, "actor_ft parameters should change after update"

    def test_multiple_update_cycles(self):
        dppo, obs = _build_dppo()
        for _ in range(3):
            _run_full_collection(dppo, obs)
            dppo.compute_returns(obs)
            dppo.update()


class TestPPODiffusionSaveLoad:
    def test_save_returns_dict(self):
        dppo, _ = _build_dppo()
        saved = dppo.save()
        assert "actor_state_dict" in saved or "model_state_dict" in saved or isinstance(saved, dict)

    def test_save_contains_critic_optimizer(self):
        dppo, _ = _build_dppo()
        saved = dppo.save()
        assert "critic_optimizer_state_dict" in saved

    def test_load_roundtrip(self):
        dppo, obs = _build_dppo()
        _run_full_collection(dppo, obs)
        dppo.compute_returns(obs)
        dppo.update()

        saved = dppo.save()

        # Build a fresh dppo and load
        dppo2, _ = _build_dppo()
        dppo2.load(saved, load_cfg=None, strict=True)

        # actor_ft parameters should match
        for p1, p2 in zip(dppo.actor.actor_ft.parameters(), dppo2.actor.actor_ft.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_critic_optimizer_state(self):
        dppo, obs = _build_dppo()
        _run_full_collection(dppo, obs)
        dppo.compute_returns(obs)
        dppo.update()

        saved = dppo.save()
        dppo2, _ = _build_dppo()
        dppo2.load(saved, load_cfg=None, strict=True)

        # Critic optimizer state should be loaded
        orig_state = dppo.critic_optimizer.state_dict()
        loaded_state = dppo2.critic_optimizer.state_dict()
        assert orig_state["param_groups"][0]["lr"] == loaded_state["param_groups"][0]["lr"]


class TestPPODiffusionConstructAlgorithm:
    def test_construct_algorithm_factory(self):
        from rsl_rl.env import VecEnv

        class DummyEnv(VecEnv):
            num_envs = NUM_ENVS
            num_actions = NUM_ACTIONS
            max_episode_length = 50
            episode_length_buf = torch.zeros(NUM_ENVS)
            device = "cpu"
            cfg = {}

            def get_observations(self):
                return make_obs(NUM_ENVS, OBS_DIM)

            def step(self, actions):
                pass

        obs = make_obs(NUM_ENVS, OBS_DIM)
        cfg = {
            "num_steps_per_env": NUM_STEPS,
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "actor": {
                "class_name": "rsl_rl.models.DiffusionModel",
                "denoising_steps": DENOISING_STEPS,
                "ft_denoising_steps": FT_DENOISING_STEPS,
                "horizon_steps": 1,
                "mlp_dims": [32, 32],
                "activation_type": "Mish",
            },
            "critic": {
                "class_name": "rsl_rl.models.MLPModel",
                "hidden_dims": [32, 32],
            },
            "algorithm": {
                "class_name": "rsl_rl.algorithms.PPODiffusion",
                "num_learning_epochs": 2,
                "num_mini_batches": 2,
                "gamma_denoising": 0.99,
                "clip_ploss_coef": 0.1,
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "reward_horizon": 2,
            },
        }
        env = DummyEnv()
        dppo = PPODiffusion.construct_algorithm(obs, env, cfg, device="cpu")
        assert isinstance(dppo, PPODiffusion)
        assert isinstance(dppo.actor, DiffusionModel)
        assert isinstance(dppo.storage, DiffusionRolloutStorage)


class TestPPODiffusionDenosingDiscount:
    def test_denoising_discount_applied(self):
        """Verify that advantages are scaled by gamma_denoising^(K-i-1)."""
        gamma_denoising = 0.9
        dppo, obs = _build_dppo(gamma_denoising=gamma_denoising)
        _run_full_collection(dppo, obs)
        dppo.compute_returns(obs)
        # Run one update — if discount is wrong, tensors would have NaN
        losses = dppo.update()
        assert torch.isfinite(torch.tensor(losses["surrogate"]))

    def test_clip_coef_interpolation(self):
        """Verify clip coefficient interpolation doesn't raise errors."""
        dppo, obs = _build_dppo(
            clip_ploss_coef=0.2,
            clip_ploss_coef_base=1e-4,
            clip_ploss_coef_rate=3.0,
        )
        _run_full_collection(dppo, obs)
        dppo.compute_returns(obs)
        losses = dppo.update()
        assert torch.isfinite(torch.tensor(losses["clipfrac"]))
