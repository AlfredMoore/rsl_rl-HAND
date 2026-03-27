"""
Diffusion model as actor for PPO, implementing the MLPModel interface from rsl_rl 5.0.1.

Ported from:
  - dppo/model/diffusion/diffusion.py (DiffusionModel)
  - dppo/model/diffusion/diffusion_vpg.py (VPGDiffusion)

The class merges both into a single DiffusionModel that can be used as the actor
in rsl_rl's PPO algorithm, duck-typing the MLPModel interface.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import namedtuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import EmpiricalNormalization, HiddenState

from rsl_rl.modules.diffusion.sampling import cosine_beta_schedule, extract, make_timesteps
from rsl_rl.modules.diffusion.denoising_network import DiffusionMLP
from rsl_rl.modules.diffusion.eta import EtaFixed

log = logging.getLogger(__name__)

Sample = namedtuple("Sample", "trajectories chains")


class DiffusionModel(nn.Module):
    """Diffusion-based actor model implementing the MLPModel interface for rsl_rl 5.0.1.

    This model uses a denoising diffusion process (DDPM/DDIM) to generate actions.
    It implements the same duck-typed interface as rsl_rl's MLPModel so that PPO
    can use it as the actor without modification.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        # Diffusion parameters
        denoising_steps: int = 100,
        horizon_steps: int = 1,
        ft_denoising_steps: int = 5,
        predict_epsilon: bool = True,
        use_ddim: bool = False,
        ddim_steps: int | None = None,
        ddim_discretize: str = "uniform",
        # Clipping
        denoised_clip_value: float = 1.0,
        randn_clip_value: float = 10.0,
        final_action_clip_value: float | None = None,
        eps_clip_value: float | None = None,
        # Denoising std
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
        # Denoising network parameters
        time_dim: int = 16,
        mlp_dims: list[int] = [256, 256],
        cond_mlp_dims: list[int] | None = None,
        activation_type: str = "Mish",
        out_activation_type: str = "Identity",
        use_layernorm: bool = False,
        residual_style: bool = False,
        # Eta parameters (for DDIM)
        eta_cfg: dict | None = None,
        learn_eta: bool = False,
        # Annealing
        ft_denoising_steps_d: int = 0,
        ft_denoising_steps_t: int = 0,
        # Obs normalization
        obs_normalization: bool = False,
        # Pre-trained checkpoint
        network_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # --- Resolve observation groups and dimensions (like MLPModel) ---
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.action_dim = output_dim
        self.horizon_steps = horizon_steps

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        # --- Diffusion parameters ---
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.denoised_clip_value = denoised_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.randn_clip_value = randn_clip_value
        self.eps_clip_value = eps_clip_value
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std

        # --- Build denoising network ---
        self.network = DiffusionMLP(
            action_dim=output_dim,
            horizon_steps=horizon_steps,
            cond_dim=self.obs_dim,
            time_dim=time_dim,
            mlp_dims=mlp_dims,
            cond_mlp_dims=cond_mlp_dims,
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
            residual_style=residual_style,
        )

        # --- DDPM schedule buffers ---
        # (copied from dppo/model/diffusion/diffusion.py DiffusionModel.__init__)
        betas = cosine_beta_schedule(denoising_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        ddpm_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("ddpm_var", ddpm_var)
        self.register_buffer("ddpm_logvar_clipped", torch.log(torch.clamp(ddpm_var, min=1e-20)))

        ddpm_mu_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ddpm_mu_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("ddpm_mu_coef1", ddpm_mu_coef1)
        self.register_buffer("ddpm_mu_coef2", ddpm_mu_coef2)

        # --- DDIM parameters ---
        # (copied from dppo/model/diffusion/diffusion.py DiffusionModel.__init__)
        if use_ddim:
            assert predict_epsilon, "DDIM requires predicting epsilon."
            if ddim_discretize == "uniform":
                step_ratio = self.denoising_steps // ddim_steps
                ddim_t = torch.arange(0, ddim_steps) * step_ratio
            else:
                raise ValueError(f"Unknown discretization: {ddim_discretize}")
            ddim_alphas = alphas_cumprod[ddim_t].clone().to(torch.float32)
            ddim_alphas_sqrt = torch.sqrt(ddim_alphas)
            ddim_alphas_prev = torch.cat([torch.tensor([1.0]).to(torch.float32), alphas_cumprod[ddim_t[:-1]]])
            ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5
            ddim_sigmas = torch.zeros_like(ddim_alphas)  # eta=0 for deterministic

            # Flip all for reverse iteration
            self.register_buffer("ddim_t", torch.flip(ddim_t, [0]))
            self.register_buffer("ddim_alphas", torch.flip(ddim_alphas, [0]))
            self.register_buffer("ddim_alphas_sqrt", torch.flip(ddim_alphas_sqrt, [0]))
            self.register_buffer("ddim_alphas_prev", torch.flip(ddim_alphas_prev, [0]))
            self.register_buffer("ddim_sqrt_one_minus_alphas", torch.flip(ddim_sqrt_one_minus_alphas, [0]))
            self.register_buffer("ddim_sigmas", torch.flip(ddim_sigmas, [0]))

        # --- VPGDiffusion: actor/actor_ft split ---
        # (copied from dppo/model/diffusion/diffusion_vpg.py VPGDiffusion.__init__)
        assert ft_denoising_steps <= denoising_steps
        if use_ddim:
            assert ft_denoising_steps <= ddim_steps

        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d
        self.ft_denoising_steps_t = ft_denoising_steps_t
        self.ft_denoising_steps_cnt = 0

        # Learnable eta for DDIM
        self.learn_eta = learn_eta
        if eta_cfg is not None:
            # Resolve eta class from config
            eta_type = eta_cfg.pop("class_name", "EtaFixed")
            from rsl_rl.modules.diffusion import eta as eta_module
            eta_class = getattr(eta_module, eta_type)
            self.eta = eta_class(**eta_cfg)
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False
        elif use_ddim:
            self.eta = EtaFixed()
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False

        # Rename network to actor, create fine-tuned copy
        self.actor = self.network
        self.actor_ft = copy.deepcopy(self.actor)
        log.info("Cloned model for fine-tuning")

        # Freeze original actor
        for param in self.actor.parameters():
            param.requires_grad = False
        log.info("Turned off gradients of the pretrained network")
        log.info(
            f"Number of finetuned parameters: {sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)}"
        )

        # Load pre-trained checkpoint if provided
        if network_path is not None:
            checkpoint = torch.load(network_path, map_location="cpu", weights_only=True)
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                log.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                log.info("Loaded RL-trained policy from %s", network_path)

        # --- State for MLPModel interface ---
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None
        self._last_chains: torch.Tensor | None = None
        self._last_obs_flat: torch.Tensor | None = None
        self._last_batch_size: int = 0

    # ========== Observation handling ==========

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"DiffusionModel only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_flat_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate and normalize selected observation groups into a flat tensor."""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        flat_obs = torch.cat(obs_list, dim=-1)
        flat_obs = self.obs_normalizer(flat_obs)
        return flat_obs

    def _make_cond(self, flat_obs: torch.Tensor) -> dict:
        """Convert flat obs tensor to dppo-style cond dict."""
        return {"state": flat_obs.unsqueeze(1)}  # (B, 1, Do)

    # ========== Diffusion sampling (from DiffusionModel + VPGDiffusion) ==========

    def p_mean_var(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict,
        index: torch.Tensor | None = None,
        use_base_policy: bool = False,
        deterministic: bool = False,
    ):
        """Compute predicted mean and variance for one denoising step.

        Copied from dppo/model/diffusion/diffusion_vpg.py VPGDiffusion.p_mean_var.
        """
        noise = self.actor(x, t, cond=cond)

        # Determine which samples are in the fine-tuning range
        if self.use_ddim:
            ft_indices = torch.where(index >= (self.ddim_steps - self.ft_denoising_steps))[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor if use_base_policy else self.actor_ft

        # Overwrite noise for fine-tuning steps
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:
            x_recon = noise

        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(cond).unsqueeze(1)  # B x 1 x (Da or 1)
            sigma = (
                etas * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = torch.log(var)
        else:
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = torch.ones_like(mu).to(mu.device)
        return mu, logvar, etas

    @torch.no_grad()
    def _sample(
        self,
        cond: dict,
        deterministic: bool = False,
        return_chain: bool = True,
        use_base_policy: bool = False,
    ) -> Sample:
        """Run the diffusion denoising loop.

        Copied from dppo/model/diffusion/diffusion_vpg.py VPGDiffusion.forward.
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:
                    std = torch.clip(std, min=1e-3)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # Clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (self.ddim_steps - self.ft_denoising_steps - 1):
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    def get_min_sampling_denoising_std(self):
        if type(self.min_sampling_denoising_std) is float:
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std()

    # ========== Log-probability computation (from VPGDiffusion) ==========

    def get_logprobs(
        self,
        cond: dict,
        chains: torch.Tensor,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """Calculate logprobs of entire chain of denoised actions.

        Copied from dppo/model/diffusion/diffusion_vpg.py VPGDiffusion.get_logprobs.

        Args:
            cond: dict with key "state": (B, To, Do)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B*K, Ta, Da)
            entropy (if get_ent): (B*K, Ta)
        """
        # Repeat cond for denoising_steps, flatten batch and time dimensions
        cond = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }

        # Repeat t for batch dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps:]
        else:
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.betas.device,
            )
        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        if self.use_ddim:
            indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.betas.device,
            )
            indices = indices_single.repeat(chains.shape[0])
        else:
            indices = None

        # Split chains
        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]

        # Flatten first two dimensions
        chains_prev = chains_prev.reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains_next.reshape(-1, self.horizon_steps, self.action_dim)

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def get_logprobs_subsample(
        self,
        cond: dict,
        chains_prev: torch.Tensor,
        chains_next: torch.Tensor,
        denoising_inds: torch.Tensor,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """Calculate logprobs of random samples of denoised chains.

        Copied from dppo/model/diffusion/diffusion_vpg.py VPGDiffusion.get_logprobs_subsample.

        Args:
            cond: dict with key "state": (B, To, Do)
            chains_prev: (B, Ta, Da)
            chains_next: (B, Ta, Da)
            denoising_inds: (B,) indices into [0, ft_denoising_steps)

        Returns:
            logprobs: (B, Ta, Da)
            entropy (if get_ent): (B, Ta)
        """
        # Sample t for batch dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps:]
        else:
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.betas.device,
            )
        t_all = t_single[denoising_inds]
        if self.use_ddim:
            ddim_indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.betas.device,
            )
            ddim_indices = ddim_indices_single[denoising_inds]
        else:
            ddim_indices = None

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    # ========== Annealing (from VPGDiffusion) ==========

    def step(self):
        """Anneal min_sampling_denoising_std and fine-tuning denoising steps.

        Copied from dppo/model/diffusion/diffusion_vpg.py VPGDiffusion.step.
        """
        if type(self.min_sampling_denoising_std) is not float:
            self.min_sampling_denoising_std.step()

        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(0, self.ft_denoising_steps - self.ft_denoising_steps_d)
            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            for param in self.actor.parameters():
                param.requires_grad = False
            log.info(f"Annealed fine-tuning denoising steps to {self.ft_denoising_steps}")

    # ========== Forward diffusion ==========

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: q(x_t | x_0).

        Copied from dppo/model/diffusion/diffusion.py DiffusionModel.q_sample.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ========== MLPModel-compatible interface ==========

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass implementing the MLPModel interface.

        Args:
            obs: TensorDict with observation groups
            masks: Unused (for recurrent compat)
            hidden_state: Unused (for recurrent compat)
            stochastic_output: If True, sample with noise and store chains

        Returns:
            actions: (B, num_actions)
        """
        flat_obs = self._get_flat_obs(obs)
        cond = self._make_cond(flat_obs)

        if stochastic_output:
            sample = self._sample(cond, deterministic=False, return_chain=True)
            actions = sample.trajectories[:, 0, :]  # (B, Da) from (B, Ta, Da) with Ta=horizon_steps
            if self.horizon_steps > 1:
                actions = sample.trajectories[:, 0, :]  # Take first step of chunk
            else:
                actions = sample.trajectories.squeeze(1)  # (B, Da)

            # Store state for get_output_log_prob
            self._last_chains = sample.chains
            self._last_obs_flat = flat_obs.detach()
            self._action_mean = actions.detach()
            self._action_std = torch.full_like(actions, self.get_min_sampling_denoising_std())
            self._last_batch_size = actions.shape[0]
        else:
            sample = self._sample(cond, deterministic=True, return_chain=False)
            if self.horizon_steps > 1:
                actions = sample.trajectories[:, 0, :]
            else:
                actions = sample.trajectories.squeeze(1)

            self._action_mean = actions.detach()
            self._action_std = torch.full_like(actions, self.get_min_sampling_denoising_std())
            self._last_batch_size = actions.shape[0]

        return actions

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups."""
        return self._get_flat_obs(obs)

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output (last denoised action)."""
        return self._action_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the std for logging compatibility."""
        return self._action_std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Entropy is not tractable for diffusion; return zeros."""
        return torch.zeros(self._last_batch_size, device=self._action_mean.device)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return (mean, std) for storage compatibility."""
        return (self._action_mean, self._action_std)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of actions using the denoising chain.

        Uses the stored chains and observations from the last forward pass
        to compute log-probabilities through the denoising process.

        Args:
            outputs: (B, Da) actions (unused — we use the stored chains)

        Returns:
            log_prob: (B,) summed log-probabilities
        """
        if self._last_chains is None or self._last_obs_flat is None:
            raise RuntimeError("Must call forward(stochastic_output=True) before get_output_log_prob")

        cond = self._make_cond(self._last_obs_flat)
        log_probs = self.get_logprobs(cond, self._last_chains, get_ent=False, use_base_policy=False)
        # log_probs shape: (B*K, Ta, Da)
        B = self._last_chains.shape[0]
        K = self.ft_denoising_steps
        log_probs = log_probs.reshape(B, K, self.horizon_steps, self.action_dim)
        # Sum over all dimensions except batch
        return log_probs.sum(dim=(1, 2, 3))

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Approximate KL divergence between old and new distributions.

        For diffusion policies, exact KL is intractable. We approximate using
        Gaussian KL on the stored mean/std.
        """
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization statistics."""
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(mlp_obs)  # type: ignore

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """No-op for non-recurrent model."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return None (non-recurrent)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """No-op for non-recurrent model."""
        pass

    def as_jit(self) -> nn.Module:
        """Return a JIT-exportable version of the model."""
        return _TorchDiffusionModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-exportable version of the model."""
        return _OnnxDiffusionModel(self, verbose)


class _TorchDiffusionModel(nn.Module):
    """Exportable diffusion model for JIT."""

    def __init__(self, model: DiffusionModel) -> None:
        super().__init__()
        self.model = copy.deepcopy(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic inference on pre-concatenated observations."""
        obs = self.model.obs_normalizer(x)
        cond = {"state": obs.unsqueeze(1)}
        sample = self.model._sample(cond, deterministic=True, return_chain=False)
        if self.model.horizon_steps > 1:
            return sample.trajectories[:, 0, :]
        return sample.trajectories.squeeze(1)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxDiffusionModel(nn.Module):
    """Exportable diffusion model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: DiffusionModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.model = copy.deepcopy(model)
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs = self.model.obs_normalizer(x)
        cond = {"state": obs.unsqueeze(1)}
        sample = self.model._sample(cond, deterministic=True, return_chain=False)
        if self.model.horizon_steps > 1:
            return sample.trajectories[:, 0, :]
        return sample.trajectories.squeeze(1)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
