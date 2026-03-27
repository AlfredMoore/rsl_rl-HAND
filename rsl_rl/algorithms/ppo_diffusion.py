"""
DPPO: Diffusion Policy Policy Optimization.

Extends rsl_rl's PPO to support diffusion-based actors with:
- Denoising chain storage and per-step credit assignment
- Denoising discount (gamma_denoising)
- Exponential clip coefficient interpolation over denoising steps
- Dual optimizers (actor_ft + eta, critic)

Loss logic ported from dppo/model/diffusion/diffusion_ppo.py PPODiffusion.loss.
"""

from __future__ import annotations

import math
import logging
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from rsl_rl.models.diffusion_model import DiffusionModel
from rsl_rl.storage.diffusion_rollout_storage import DiffusionRolloutStorage

log = logging.getLogger(__name__)


class PPODiffusion(PPO):
    """PPO algorithm extended for diffusion-based actors.

    The actor is a DiffusionModel that generates actions via iterative denoising.
    The critic is a standard MLPModel. Training uses per-denoising-step credit
    assignment with gamma_denoising discount and exponential clip interpolation.
    """

    actor: DiffusionModel

    def __init__(
        self,
        actor: DiffusionModel,
        critic: MLPModel,
        storage: DiffusionRolloutStorage,
        # Diffusion PPO specific
        gamma_denoising: float = 0.99,
        clip_ploss_coef: float = 0.1,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3.0,
        clip_vloss_coef: float | None = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        norm_adv: bool = True,
        reward_horizon: int = 4,
        # Dual optimizer learning rates
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-3,
        # Standard PPO params (passed to parent)
        **kwargs,
    ) -> None:
        # Initialize parent PPO (creates self.optimizer over actor+critic)
        # We'll override the optimizer below
        super().__init__(actor=actor, critic=critic, storage=storage, **kwargs)

        # Diffusion-specific parameters
        # (from dppo/model/diffusion/diffusion_ppo.py PPODiffusion.__init__)
        self.gamma_denoising = gamma_denoising
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        self.norm_adv = norm_adv
        self.reward_horizon = reward_horizon

        # Override with dual optimizers
        actor_params = list(self.actor.actor_ft.parameters())
        if hasattr(self.actor, "eta") and self.actor.learn_eta:
            actor_params += list(self.actor.eta.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Keep self.optimizer pointing to actor_optimizer for runner save/load compat
        self.optimizer = self.actor_optimizer

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data including diffusion chains."""
        # Record hidden states
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())

        # Forward actor (stochastic)
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()

        # Forward critic
        self.transition.values = self.critic(obs).detach()

        # Log-prob and distribution params
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)

        # Store observations
        self.transition.observations = obs

        # Store diffusion-specific data (chains and per-step log-probs)
        # These will be added to storage after the standard add_transition
        self._pending_chains = self.actor._last_chains.detach()
        self._pending_denoising_logprobs = self._compute_denoising_logprobs().detach()

        return self.transition.actions

    def _compute_denoising_logprobs(self) -> torch.Tensor:
        """Compute per-denoising-step log-probabilities from stored chains.

        Returns:
            (num_envs, K, Ta, Da) per-step log-probs
        """
        cond = self.actor._make_cond(self.actor._last_obs_flat)
        log_probs = self.actor.get_logprobs(cond, self.actor._last_chains, get_ent=False)
        B = self.actor._last_chains.shape[0]
        K = self.actor.ft_denoising_steps
        return log_probs.reshape(B, K, self.actor.horizon_steps, self.actor.action_dim)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record environment step, including diffusion chain data."""
        # Update normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        # Standard transition processing
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Store the step index before add_transition increments it
        step_idx = self.storage.step

        # Record the transition (increments self.storage.step)
        self.storage.add_transition(self.transition)

        # Store diffusion chain data at the same step
        self.storage.add_diffusion_data(step_idx, self._pending_chains, self._pending_denoising_logprobs)

        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def update(self) -> dict[str, float]:
        """Run optimization using diffusion-specific PPO loss.

        Uses the diffusion mini-batch generator which provides chain data.
        Loss logic ported from dppo/model/diffusion/diffusion_ppo.py PPODiffusion.loss.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_approx_kl = 0.0
        mean_clipfrac = 0.0

        generator = self.storage.diffusion_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for batch in generator:
            # --- Forward actor on batch observations to set up internal state ---
            self.actor(batch.observations, stochastic_output=True)

            # --- Build cond dict from batch observations ---
            flat_obs = self.actor._get_flat_obs(batch.observations)
            cond = self.actor._make_cond(flat_obs)

            # --- Get new log-probs for the sampled denoising step ---
            # (from dppo/model/diffusion/diffusion_ppo.py PPODiffusion.loss)
            newlogprobs, eta = self.actor.get_logprobs_subsample(
                cond,
                batch.chains_prev,
                batch.chains_next,
                batch.denoising_inds,
                get_ent=True,
            )
            entropy_loss = -eta.mean()
            newlogprobs = newlogprobs.clamp(min=-5, max=2)

            # Old log-probs from storage
            oldlogprobs = batch.old_denoising_logprobs.clamp(min=-5, max=2)

            # Only backpropagate through reward_horizon steps
            newlogprobs = newlogprobs[:, :self.reward_horizon, :]
            oldlogprobs = oldlogprobs[:, :self.reward_horizon, :]

            # Average over action and time dimensions
            newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
            oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)

            # --- Advantage processing ---
            advantages = torch.squeeze(batch.advantages)
            if self.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Clip advantages by quantiles
            advantage_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
            advantage_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
            advantages = advantages.clamp(min=advantage_min, max=advantage_max)

            # Denoising discount
            discount = torch.tensor(
                [self.gamma_denoising ** (self.actor.ft_denoising_steps - i - 1) for i in batch.denoising_inds],
                device=self.device,
            )
            advantages = advantages * discount

            # --- PPO ratio ---
            logratio = newlogprobs - oldlogprobs
            ratio = logratio.exp()

            # Exponential clip interpolation over denoising steps
            t = (batch.denoising_inds.float() / max(self.actor.ft_denoising_steps - 1, 1)).to(self.device)
            if self.actor.ft_denoising_steps > 1:
                clip_coef = self.clip_ploss_coef_base + (
                    self.clip_ploss_coef - self.clip_ploss_coef_base
                ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                    math.exp(self.clip_ploss_coef_rate) - 1
                )
            else:
                clip_coef = t

            # KL and clip statistics
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

            # Policy loss with clipping
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # --- Value loss ---
            values = self.critic(batch.observations)
            returns = batch.returns
            if self.clip_vloss_coef is not None:
                v_loss_unclipped = (values - returns) ** 2
                v_clipped = batch.values + torch.clamp(
                    values - batch.values, -self.clip_vloss_coef, self.clip_vloss_coef
                )
                v_loss_clipped = (v_clipped - returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((values - returns) ** 2).mean()

            # --- Backward and step ---
            # Actor optimizer
            actor_loss = pg_loss + entropy_loss * self.entropy_coef
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.actor_ft.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Critic optimizer
            critic_loss = self.value_loss_coef * v_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # Accumulate for logging
            mean_value_loss += v_loss.item()
            mean_surrogate_loss += pg_loss.item()
            mean_entropy += entropy_loss.item()
            mean_approx_kl += approx_kl.item()
            mean_clipfrac += clipfrac

        # Call annealing step on actor
        self.actor.step()

        # Average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_approx_kl /= num_updates
        mean_clipfrac /= num_updates

        # Clear storage
        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "approx_kl": mean_approx_kl,
            "clipfrac": mean_clipfrac,
        }

    def save(self) -> dict:
        """Return dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load models from a saved dict."""
        result = super().load(loaded_dict, load_cfg, strict)
        if "critic_optimizer_state_dict" in loaded_dict:
            if load_cfg is None or load_cfg.get("optimizer", True):
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        return result

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPODiffusion:
        """Construct the PPODiffusion algorithm from config.

        Expected config structure:
            cfg["actor"]["class_name"]: "rsl_rl_diffusion.models.DiffusionModel"
            cfg["critic"]["class_name"]: "MLPModel" (or qualified name)
            cfg["algorithm"]["class_name"]: "rsl_rl_diffusion.algorithms.PPODiffusion"
            cfg["algorithm"]["gamma_denoising"]: float
            cfg["algorithm"]["clip_ploss_coef"]: float
            cfg["algorithm"]["actor_lr"]: float
            cfg["algorithm"]["critic_lr"]: float
            ...
        """
        # Resolve class callables
        alg_class: type[PPODiffusion] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[DiffusionModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Create actor (DiffusionModel)
        actor: DiffusionModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")

        # Create critic (standard MLPModel)
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")

        # Create storage (DiffusionRolloutStorage)
        storage = DiffusionRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            ft_denoising_steps=cfg["actor"].get("ft_denoising_steps", 5),
            horizon_steps=cfg["actor"].get("horizon_steps", 1),
            action_dim=env.num_actions,
        )

        # Create algorithm
        alg: PPODiffusion = alg_class(
            actor, critic, storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )

        return alg
