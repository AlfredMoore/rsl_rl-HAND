"""
Extended rollout storage for diffusion policy training.

Stores additional buffers for denoising chains and per-step log-probabilities
required by the PPODiffusion algorithm.
"""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from .rollout_storage import RolloutStorage


class DiffusionRolloutStorage(RolloutStorage):
    """Rollout storage extended with diffusion chain buffers.

    In addition to the standard rollout data, this stores:
    - chains: the denoising chain at each environment step
    - denoising_logprobs: per-step log-probabilities along the chain
    """

    class DiffusionTransition(RolloutStorage.Transition):
        """Transition with additional diffusion chain data."""

        def __init__(self) -> None:
            super().__init__()
            self.chains: torch.Tensor | None = None
            self.denoising_logprobs: torch.Tensor | None = None

        def clear(self) -> None:
            super().clear()
            self.chains = None
            self.denoising_logprobs = None

    class DiffusionBatch(RolloutStorage.Batch):
        """Batch with additional diffusion chain data."""

        def __init__(
            self,
            chains_prev: torch.Tensor | None = None,
            chains_next: torch.Tensor | None = None,
            denoising_inds: torch.Tensor | None = None,
            old_denoising_logprobs: torch.Tensor | None = None,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.chains_prev = chains_prev
            self.chains_next = chains_next
            self.denoising_inds = denoising_inds
            self.old_denoising_logprobs = old_denoising_logprobs

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        # Diffusion-specific parameters
        ft_denoising_steps: int = 5,
        horizon_steps: int = 1,
        action_dim: int | None = None,
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)

        if action_dim is None:
            action_dim = actions_shape[0] if isinstance(actions_shape, (list, tuple)) else actions_shape

        self.ft_denoising_steps = ft_denoising_steps
        self.horizon_steps = horizon_steps
        self._action_dim = action_dim

        # Additional buffers for diffusion chains
        # chains: (num_transitions, num_envs, K+1, Ta, Da)
        self.chains = torch.zeros(
            num_transitions_per_env, num_envs,
            ft_denoising_steps + 1, horizon_steps, action_dim,
            device=device,
        )
        # denoising_logprobs: (num_transitions, num_envs, K, Ta, Da)
        self.denoising_logprobs = torch.zeros(
            num_transitions_per_env, num_envs,
            ft_denoising_steps, horizon_steps, action_dim,
            device=device,
        )

    def add_diffusion_data(self, step: int, chains: torch.Tensor, denoising_logprobs: torch.Tensor) -> None:
        """Store diffusion chain data for a given step.

        Args:
            step: The step index (should match self.step - 1 after add_transition)
            chains: (num_envs, K+1, Ta, Da)
            denoising_logprobs: (num_envs, K, Ta, Da)
        """
        self.chains[step].copy_(chains)
        self.denoising_logprobs[step].copy_(denoising_logprobs)

    def diffusion_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[DiffusionBatch, None, None]:
        """Yield mini-batches with diffusion chain data for PPODiffusion training.

        Each sample in the batch corresponds to one (env_step, env, denoising_step)
        triple. The generator samples uniformly over env_steps and envs, and
        randomly selects a denoising index for each sample.

        Yields DiffusionBatch objects with:
            - Standard PPO fields (observations, actions, advantages, etc.)
            - chains_prev: (batch_size, Ta, Da) - chain state before denoising step
            - chains_next: (batch_size, Ta, Da) - chain state after denoising step
            - denoising_inds: (batch_size,) - which denoising step [0, K)
            - old_denoising_logprobs: (batch_size, Ta, Da) - old log-probs at that step
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for RL training.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        # Flatten standard buffers: (T, E) -> (T*E)
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)

        # Flatten chain buffers: (T, E, ...) -> (T*E, ...)
        chains = self.chains.flatten(0, 1)  # (T*E, K+1, Ta, Da)
        denoising_logprobs = self.denoising_logprobs.flatten(0, 1)  # (T*E, K, Ta, Da)

        for epoch in range(num_epochs):
            # Random permutation over (env_step * num_envs)
            indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
            # Random denoising indices for each sample
            denoising_inds_all = torch.randint(
                0, self.ft_denoising_steps, (num_mini_batches * mini_batch_size,),
                device=self.device,
            )

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                denoising_inds = denoising_inds_all[start:stop]

                # Extract chain pairs for the selected denoising indices
                batch_chains = chains[batch_idx]  # (mb, K+1, Ta, Da)
                chains_prev = batch_chains[
                    torch.arange(mini_batch_size, device=self.device),
                    denoising_inds,
                ]  # (mb, Ta, Da)
                chains_next = batch_chains[
                    torch.arange(mini_batch_size, device=self.device),
                    denoising_inds + 1,
                ]  # (mb, Ta, Da)

                # Extract old log-probs for the selected denoising indices
                batch_denoising_lp = denoising_logprobs[batch_idx]  # (mb, K, Ta, Da)
                old_den_lp = batch_denoising_lp[
                    torch.arange(mini_batch_size, device=self.device),
                    denoising_inds,
                ]  # (mb, Ta, Da)

                yield DiffusionRolloutStorage.DiffusionBatch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    chains_prev=chains_prev,
                    chains_next=chains_next,
                    denoising_inds=denoising_inds,
                    old_denoising_logprobs=old_den_lp,
                )
