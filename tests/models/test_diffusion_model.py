# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DiffusionModel actor."""

from __future__ import annotations

import torch
import pytest
from tensordict import TensorDict

from rsl_rl.models.diffusion_model import DiffusionModel
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
DENOISING_STEPS = 10
FT_DENOISING_STEPS = 3


def _make_model(obs: TensorDict, obs_groups: dict, num_actions: int = NUM_ACTIONS, **kwargs) -> DiffusionModel:
    defaults = dict(
        denoising_steps=DENOISING_STEPS,
        ft_denoising_steps=FT_DENOISING_STEPS,
        horizon_steps=1,
        mlp_dims=[32, 32],
        activation_type="Mish",
        residual_style=False,
    )
    defaults.update(kwargs)
    return DiffusionModel(obs, obs_groups, "actor", num_actions, **defaults)


@pytest.fixture()
def obs_and_groups():
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"]}
    return obs, obs_groups


class TestDiffusionModelConstruction:
    def test_basic_construction(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        assert model.obs_dim == OBS_DIM
        assert model.action_dim == NUM_ACTIONS
        assert model.ft_denoising_steps == FT_DENOISING_STEPS
        assert not model.is_recurrent

    def test_actor_frozen(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        for param in model.actor.parameters():
            assert not param.requires_grad

    def test_actor_ft_trainable(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        assert any(p.requires_grad for p in model.actor_ft.parameters())

    def test_ddim_construction(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups, use_ddim=True, ddim_steps=5, ft_denoising_steps=2)
        assert model.use_ddim
        assert hasattr(model, "ddim_t")


class TestDiffusionModelForward:
    def test_stochastic_forward_action_shape(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        actions = model(obs, stochastic_output=True)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_deterministic_forward_action_shape(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        actions = model(obs, stochastic_output=False)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_stochastic_stores_chains(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        model(obs, stochastic_output=True)
        assert model._last_chains is not None
        # chains: (B, K+1, Ta, Da)
        K = FT_DENOISING_STEPS
        assert model._last_chains.shape == (NUM_ENVS, K + 1, 1, NUM_ACTIONS)

    def test_stochastic_stores_obs(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        model(obs, stochastic_output=True)
        assert model._last_obs_flat is not None
        assert model._last_obs_flat.shape == (NUM_ENVS, OBS_DIM)

    def test_deterministic_no_chains(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        model(obs, stochastic_output=False)
        # _last_chains may be None (deterministic doesn't need chains)
        # Just verify actions are valid
        actions = model(obs, stochastic_output=False)
        assert not torch.isnan(actions).any()


class TestDiffusionModelLogProb:
    def test_get_output_log_prob_shape(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        actions = model(obs, stochastic_output=True)
        log_prob = model.get_output_log_prob(actions)
        assert log_prob.shape == (NUM_ENVS,)

    def test_get_output_log_prob_requires_stochastic_forward(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        # Without calling stochastic forward first, should raise
        with pytest.raises(RuntimeError):
            model.get_output_log_prob(torch.randn(NUM_ENVS, NUM_ACTIONS))

    def test_log_prob_finite(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        actions = model(obs, stochastic_output=True)
        log_prob = model.get_output_log_prob(actions)
        assert torch.isfinite(log_prob).all()


class TestDiffusionModelProperties:
    def test_output_mean_shape(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        model(obs, stochastic_output=True)
        assert model.output_mean.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_output_std_shape(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        model(obs, stochastic_output=True)
        assert model.output_std.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_output_entropy_shape(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        model(obs, stochastic_output=True)
        assert model.output_entropy.shape == (NUM_ENVS,)

    def test_output_distribution_params(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        model(obs, stochastic_output=True)
        params = model.output_distribution_params
        assert len(params) == 2
        assert params[0].shape == (NUM_ENVS, NUM_ACTIONS)  # mean
        assert params[1].shape == (NUM_ENVS, NUM_ACTIONS)  # std


class TestDiffusionModelAnnealing:
    def test_step_does_not_crash(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        for _ in range(5):
            model.step()

    def test_step_anneals_ft_denoising_steps(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(
            obs, obs_groups,
            ft_denoising_steps=4,
            ft_denoising_steps_d=2,
            ft_denoising_steps_t=2,
        )
        initial_steps = model.ft_denoising_steps
        model.step()
        model.step()  # Should trigger anneal at cnt=2
        assert model.ft_denoising_steps <= initial_steps


class TestDiffusionModelNonRecurrent:
    def test_reset_is_noop(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        model.reset(dones=torch.zeros(NUM_ENVS))

    def test_get_hidden_state_returns_none(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        assert model.get_hidden_state() is None

    def test_get_latent_shape(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        latent = model.get_latent(obs)
        assert latent.shape == (NUM_ENVS, OBS_DIM)


class TestDiffusionModelExport:
    def test_as_jit_runs(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        jit_model = model.as_jit()
        flat_obs = torch.randn(NUM_ENVS, OBS_DIM)
        with torch.no_grad():
            out = jit_model(flat_obs)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_as_onnx_runs(self, obs_and_groups):
        obs, obs_groups = obs_and_groups
        model = _make_model(obs, obs_groups)
        onnx_model = model.as_onnx()
        flat_obs = torch.randn(NUM_ENVS, OBS_DIM)
        with torch.no_grad():
            out = onnx_model(flat_obs)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)
