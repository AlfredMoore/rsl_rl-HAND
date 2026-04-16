# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DiffusionRolloutStorage."""

from __future__ import annotations

import torch
import pytest
from tensordict import TensorDict

from rsl_rl.storage.diffusion_rollout_storage import DiffusionRolloutStorage
from rsl_rl.storage.rollout_storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
NUM_STEPS = 8
FT_DENOISING_STEPS = 3
HORIZON_STEPS = 1


def _make_storage(**kwargs) -> DiffusionRolloutStorage:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    defaults = dict(
        ft_denoising_steps=FT_DENOISING_STEPS,
        horizon_steps=HORIZON_STEPS,
        action_dim=NUM_ACTIONS,
    )
    defaults.update(kwargs)
    return DiffusionRolloutStorage(
        "rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], **defaults
    )


class TestDiffusionRolloutStorageConstruction:
    def test_chains_buffer_shape(self):
        storage = _make_storage()
        expected = (NUM_STEPS, NUM_ENVS, FT_DENOISING_STEPS + 1, HORIZON_STEPS, NUM_ACTIONS)
        assert storage.chains.shape == expected

    def test_denoising_logprobs_buffer_shape(self):
        storage = _make_storage()
        expected = (NUM_STEPS, NUM_ENVS, FT_DENOISING_STEPS, HORIZON_STEPS, NUM_ACTIONS)
        assert storage.denoising_logprobs.shape == expected

    def test_inherits_rollout_storage(self):
        storage = _make_storage()
        assert isinstance(storage, RolloutStorage)


class TestAddDiffusionData:
    def test_add_diffusion_data_stores_values(self):
        storage = _make_storage()
        chains = torch.randn(NUM_ENVS, FT_DENOISING_STEPS + 1, HORIZON_STEPS, NUM_ACTIONS)
        logprobs = torch.randn(NUM_ENVS, FT_DENOISING_STEPS, HORIZON_STEPS, NUM_ACTIONS)
        step = 0
        storage.add_diffusion_data(step, chains, logprobs)
        assert torch.allclose(storage.chains[step], chains)
        assert torch.allclose(storage.denoising_logprobs[step], logprobs)

    def test_add_diffusion_data_multiple_steps(self):
        storage = _make_storage()
        for step in range(NUM_STEPS):
            chains = torch.full(
                (NUM_ENVS, FT_DENOISING_STEPS + 1, HORIZON_STEPS, NUM_ACTIONS), float(step)
            )
            logprobs = torch.full(
                (NUM_ENVS, FT_DENOISING_STEPS, HORIZON_STEPS, NUM_ACTIONS), float(step)
            )
            storage.add_diffusion_data(step, chains, logprobs)

        assert torch.allclose(storage.chains[0], torch.zeros_like(storage.chains[0]))
        assert torch.allclose(
            storage.chains[NUM_STEPS - 1],
            torch.full_like(storage.chains[NUM_STEPS - 1], float(NUM_STEPS - 1)),
        )


def _fill_storage(storage: DiffusionRolloutStorage) -> None:
    """Fill storage with dummy transitions and diffusion data."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    for step in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
        t.values = torch.randn(NUM_ENVS, 1)
        t.actions_log_prob = torch.randn(NUM_ENVS)
        t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        step_idx = storage.step
        storage.add_transition(t)
        chains = torch.randn(NUM_ENVS, FT_DENOISING_STEPS + 1, HORIZON_STEPS, NUM_ACTIONS)
        logprobs = torch.randn(NUM_ENVS, FT_DENOISING_STEPS, HORIZON_STEPS, NUM_ACTIONS)
        storage.add_diffusion_data(step_idx, chains, logprobs)

    # Compute dummy returns/advantages
    storage.returns = torch.randn(NUM_STEPS, NUM_ENVS, 1)
    storage.advantages = torch.randn(NUM_STEPS, NUM_ENVS, 1)


class TestDiffusionMiniBatchGenerator:
    def test_yields_correct_batch_size(self):
        storage = _make_storage()
        _fill_storage(storage)
        num_mini_batches = 2
        mini_batch_size = NUM_ENVS * NUM_STEPS // num_mini_batches
        batches = list(storage.diffusion_mini_batch_generator(num_mini_batches, num_epochs=1))
        assert len(batches) == num_mini_batches
        for batch in batches:
            assert batch.observations["policy"].shape[0] == mini_batch_size

    def test_chains_prev_shape(self):
        storage = _make_storage()
        _fill_storage(storage)
        num_mini_batches = 2
        mini_batch_size = NUM_ENVS * NUM_STEPS // num_mini_batches
        for batch in storage.diffusion_mini_batch_generator(num_mini_batches, num_epochs=1):
            assert batch.chains_prev.shape == (mini_batch_size, HORIZON_STEPS, NUM_ACTIONS)
            assert batch.chains_next.shape == (mini_batch_size, HORIZON_STEPS, NUM_ACTIONS)

    def test_denoising_inds_range(self):
        storage = _make_storage()
        _fill_storage(storage)
        for batch in storage.diffusion_mini_batch_generator(2, num_epochs=1):
            assert batch.denoising_inds.min() >= 0
            assert batch.denoising_inds.max() < FT_DENOISING_STEPS

    def test_old_denoising_logprobs_shape(self):
        storage = _make_storage()
        _fill_storage(storage)
        num_mini_batches = 2
        mini_batch_size = NUM_ENVS * NUM_STEPS // num_mini_batches
        for batch in storage.diffusion_mini_batch_generator(num_mini_batches, num_epochs=1):
            assert batch.old_denoising_logprobs.shape == (mini_batch_size, HORIZON_STEPS, NUM_ACTIONS)

    def test_multiple_epochs(self):
        storage = _make_storage()
        _fill_storage(storage)
        num_epochs = 3
        num_mini_batches = 2
        batches = list(storage.diffusion_mini_batch_generator(num_mini_batches, num_epochs=num_epochs))
        assert len(batches) == num_mini_batches * num_epochs

    def test_requires_rl_training_type(self):
        obs = make_obs(NUM_ENVS, OBS_DIM)
        storage = DiffusionRolloutStorage(
            "distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS],
            ft_denoising_steps=FT_DENOISING_STEPS,
            horizon_steps=HORIZON_STEPS,
            action_dim=NUM_ACTIONS,
        )
        with pytest.raises(ValueError):
            list(storage.diffusion_mini_batch_generator(2, num_epochs=1))

    def test_chains_prev_next_are_consecutive(self):
        """chains_prev and chains_next should be consecutive elements of the denoising chain."""
        storage = _make_storage()
        obs = make_obs(NUM_ENVS, OBS_DIM)
        # Fill with identifiable chain data
        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            step_idx = storage.step
            storage.add_transition(t)

            # Chain[i] = float(i) for each denoising index
            chains = torch.stack(
                [torch.full((NUM_ENVS, HORIZON_STEPS, NUM_ACTIONS), float(k))
                 for k in range(FT_DENOISING_STEPS + 1)],
                dim=1,
            )
            logprobs = torch.randn(NUM_ENVS, FT_DENOISING_STEPS, HORIZON_STEPS, NUM_ACTIONS)
            storage.add_diffusion_data(step_idx, chains, logprobs)

        storage.returns = torch.randn(NUM_STEPS, NUM_ENVS, 1)
        storage.advantages = torch.randn(NUM_STEPS, NUM_ENVS, 1)

        for batch in storage.diffusion_mini_batch_generator(2, num_epochs=1):
            inds = batch.denoising_inds
            # For each sample, chains_prev[i] == chains[denoising_inds[i]] and
            # chains_next[i] == chains[denoising_inds[i]+1]
            # With our known data: chains_prev values should equal float(denoising_ind)
            for i, idx in enumerate(inds):
                assert torch.allclose(
                    batch.chains_prev[i],
                    torch.full((HORIZON_STEPS, NUM_ACTIONS), float(idx)),
                )
                assert torch.allclose(
                    batch.chains_next[i],
                    torch.full((HORIZON_STEPS, NUM_ACTIONS), float(idx + 1)),
                )
