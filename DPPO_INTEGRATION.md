# DPPO Integration into rsl_rl 5.0.1

## Overview

This branch (`dppo_5.0.1`) extends rsl-rl-lib 5.0.1 with Diffusion Policy Policy Optimization (DPPO). The diffusion model serves as the actor (replacing Gaussian MLP), while the critic remains a standard `MLPModel`.

Ported from: `dppo/` (https://github.com/BerkeleyAutomation/dppo)

## Added Files

### Diffusion Building Blocks — `rsl_rl/modules/diffusion/`

| File | Contents | Source |
|---|---|---|
| `sampling.py` | `cosine_beta_schedule`, `extract`, `make_timesteps` | `dppo/model/diffusion/sampling.py` |
| `mlp.py` | `MLP`, `ResidualMLP`, `TwoLayerPreActivationResNetLinear` | `dppo/model/common/mlp.py` |
| `sinusoidal_emb.py` | `SinusoidalPosEmb` | `dppo/model/diffusion/modules.py` |
| `denoising_network.py` | `DiffusionMLP` | `dppo/model/diffusion/mlp_diffusion.py` |
| `eta.py` | `EtaFixed`, `EtaAction`, `EtaState`, `EtaStateAction` | `dppo/model/diffusion/eta.py` |
| `__init__.py` | Re-exports all building blocks | — |

### Model — `rsl_rl/models/diffusion_model.py`

`DiffusionModel(nn.Module)` — duck-types the `MLPModel` interface so PPO/runner can use it as actor without modification.

**Sources merged**: `dppo/model/diffusion/diffusion.py` (DiffusionModel) + `dppo/model/diffusion/diffusion_vpg.py` (VPGDiffusion)

**Key design:**
- DDPM/DDIM denoising loop generates actions
- Frozen `self.actor` handles early denoising steps; trainable `self.actor_ft` (deep copy) handles last `ft_denoising_steps`
- `forward(obs, masks, hidden_state, stochastic_output)` matches MLPModel signature
- Properties `output_mean`, `output_std`, `output_entropy`, `output_distribution_params` for runner logging
- `get_output_log_prob(actions)` computes log-probs through the denoising chain
- `get_logprobs_subsample(cond, chains_prev, chains_next, denoising_inds)` for per-step training
- `step()` for annealing `ft_denoising_steps` and `min_sampling_denoising_std`
- `as_jit()` / `as_onnx()` for export
- Supports `network_path` to load pre-trained SL checkpoint

**Constructor key params:**
- `denoising_steps` (default 100): total DDPM steps
- `ft_denoising_steps` (default 5): number of steps fine-tuned by RL
- `horizon_steps` (default 1): action chunk length
- `use_ddim` / `ddim_steps`: use DDIM sampling
- `mlp_dims`, `time_dim`, `activation_type`, `residual_style`: network architecture
- `eta_cfg`, `learn_eta`: learnable DDIM eta
- `ft_denoising_steps_d`, `ft_denoising_steps_t`: annealing schedule

### Storage — `rsl_rl/storage/diffusion_rollout_storage.py`

`DiffusionRolloutStorage(RolloutStorage)` — adds buffers for denoising chains and per-step log-probs.

**Additional buffers:**
- `chains`: `(T, E, K+1, Ta, Da)` — full denoising chain per env step
- `denoising_logprobs`: `(T, E, K, Ta, Da)` — per-step log-probs

**Key method:** `diffusion_mini_batch_generator(num_mini_batches, num_epochs)` yields `DiffusionBatch` objects with `chains_prev`, `chains_next`, `denoising_inds`, `old_denoising_logprobs`.

### Algorithm — `rsl_rl/algorithms/ppo_diffusion.py`

`PPODiffusion(PPO)` — extends PPO with diffusion-specific training.

**Source**: `dppo/model/diffusion/diffusion_ppo.py`

**Key differences from standard PPO:**
- **Dual optimizers**: `actor_optimizer` (actor_ft + eta params), `critic_optimizer`
- **`act()`**: stores denoising chains and per-step log-probs alongside standard transition
- **`update()`**: uses denoising discount `gamma_denoising^(K-i-1)`, exponential clip coefficient interpolation, per-step credit assignment. Calls `actor.step()` for annealing.
- **`construct_algorithm()`**: factory creates `DiffusionModel` actor + `MLPModel` critic + `DiffusionRolloutStorage`

**Constructor key params:**
- `gamma_denoising` (default 0.99): discount over denoising steps
- `clip_ploss_coef` (default 0.1): max clip coefficient
- `clip_ploss_coef_base` (default 1e-3): min clip coefficient
- `clip_ploss_coef_rate` (default 3.0): exponential interpolation rate
- `clip_vloss_coef`: optional value loss clipping
- `reward_horizon` (default 4): number of denoising steps to backprop through
- `actor_lr` (default 1e-5), `critic_lr` (default 1e-3)

## Modified Files

- `rsl_rl/models/__init__.py` — added `DiffusionModel` export
- `rsl_rl/algorithms/__init__.py` — added `PPODiffusion` export
- `rsl_rl/storage/__init__.py` — added `DiffusionRolloutStorage` export

## Usage

### Installation

```bash
pip install -e /path/to/rsl_rl-HAND
```

### IsaacLab Config Example

```python
cfg = {
    "actor": {
        "class_name": "rsl_rl.models.DiffusionModel",
        "denoising_steps": 100,
        "ft_denoising_steps": 5,
        "horizon_steps": 1,
        "mlp_dims": [256, 256],
        "activation_type": "Mish",
        "residual_style": False,
        "network_path": "/path/to/pretrained.pt",  # optional SL checkpoint
    },
    "critic": {
        "class_name": "rsl_rl.models.MLPModel",
        # standard MLPModel config
    },
    "algorithm": {
        "class_name": "rsl_rl.algorithms.PPODiffusion",
        "gamma_denoising": 0.99,
        "clip_ploss_coef": 0.1,
        "clip_ploss_coef_base": 1e-3,
        "clip_ploss_coef_rate": 3.0,
        "reward_horizon": 4,
        "actor_lr": 1e-5,
        "critic_lr": 1e-3,
    },
    # ... standard PPO params (gamma, lam, num_learning_epochs, etc.)
}
```

The runner's `construct_algorithm()` pattern handles everything — no runner modifications needed.

## Porting Principle

All dppo source code was **copied faithfully first**, then **minimally modified** for rsl_rl 5.0.1 integration. Changes were limited to:
- Import paths (`dppo.model.diffusion.*` → `rsl_rl.modules.diffusion.*`)
- Observation handling (`TensorDict` → flat tensor via `_get_flat_obs()` → dppo-style `cond` dict via `_make_cond()`)
- Interface adaptation (MLPModel duck-typing: `forward` signature, properties, `get_output_log_prob`)
- No dppo logic was rewritten or generated from memory

## Test Environment

- conda env: `dppo-test` (Python 3.11)
- Verified: imports, shape tests, full algorithm pipeline (rollout → compute_returns → update → save/load)
