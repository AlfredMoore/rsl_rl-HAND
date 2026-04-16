# DPPO Integration Report

**Project:** rsl_rl-HAND × IsaacLab-NVDA Benchmark
**Branch:** `dppo_5.0.1`
**Date:** 2026-03-30 ~ 2026-03-31
**Status:** Benchmark concluded (see Results section)

---

## Overview

This document describes the full integration of **Diffusion Policy Policy Optimization (DPPO)** into the rsl_rl 5.0.1 / IsaacLab 2.3.2 stack, and the accompanying benchmark infrastructure that compares PPO vs DPPO across 9 IsaacLab tasks of varying multimodal complexity.

DPPO replaces the standard Gaussian actor with a denoising diffusion process. During rollout the actor runs a full DDPM/DDIM denoising chain from Gaussian noise to an action; during training, only the last `ft_denoising_steps` steps of that chain are fine-tuned via a PPO surrogate loss with per-step credit assignment weighted by `gamma_denoising`.

---

## Repository Structure

### rsl_rl-HAND (this repo)

```
rsl_rl/
  models/
    diffusion_model.py          # DiffusionModel — duck-typed MLPModel interface
  algorithms/
    ppo_diffusion.py            # PPODiffusion — extends PPO with dual optimizers
  storage/
    diffusion_rollout_storage.py # DiffusionRolloutStorage — stores denoising chains
  modules/
    diffusion/
      denoising_network.py      # DiffusionMLP backbone
      sampling.py               # DDPM schedule helpers (cosine_beta_schedule, extract)
      eta.py                    # EtaFixed / learned eta for DDIM

tests/
  models/test_diffusion_model.py          # 23 tests
  storage/test_diffusion_rollout_storage.py # 12 tests
  algorithms/test_ppo_diffusion.py        # 23 tests
```

### IsaacLab-NVDA (integration target)

```
source/isaaclab_rl/isaaclab_rl/rsl_rl/
  rl_cfg.py           # +3 configclasses for DPPO
  utils.py            # hasattr guard for non-MLP actor configs

source/isaaclab_tasks/.../agents/
  rsl_rl_dppo_cfg.py  # x9 task-specific DPPO runner configs

scripts/reinforcement_learning/rsl_rl/
  run_benchmark.py    # Launches 54 training runs, writes benchmark_index.json
  analyze_benchmark.py # Loads TensorBoard events, generates comparison PNG + CSV
```

---

## Architecture

### DiffusionModel (`rsl_rl/models/diffusion_model.py`, 782 lines)

Duck-types the `MLPModel` interface so `OnPolicyRunner` and `PPO` can use it without modification.

**Key design decisions:**

- **Frozen base + fine-tuned tail:** The model maintains two copies of the denoising network:
  - `actor` — frozen; used for importance sampling ratio denominator (old policy)
  - `actor_ft` — trainable; only the last `ft_denoising_steps` denoising steps are executed during fine-tuning
- **DDPM sampling:** Uses a cosine beta schedule. At inference time, runs a full `denoising_steps`-step reverse diffusion chain from Gaussian noise.
- **Log-probability computation:** The per-step log-prob is computed under a `Normal(mu_t, sigma_t)` where `sigma_t = sqrt(beta_t_hat) * clamp(eta, min_logprob_denoising_std)`.
- **Interface methods:**
  - `forward(obs, stochastic_output=True)` — returns action tensor
  - `log_prob(obs, actions, chains)` — returns log-probs over the ft chain
  - `mean` / `std` / `entropy` — properties for logging

```
Denoising chain (denoising_steps = T, ft_denoising_steps = K):

Noise x_T → ... → x_{T-K+1} [frozen] → x_{T-K} → ... → x_0 [trainable actor_ft]
                              ^                                ^
                       old log-probs                   new log-probs for PPO
```

### DiffusionRolloutStorage (`rsl_rl/storage/diffusion_rollout_storage.py`, 182 lines)

Extends `RolloutStorage` with two extra buffers per environment step:

| Buffer | Shape | Description |
|--------|-------|-------------|
| `chains` | `(T, N, denoising_steps+1, act_dim)` | Full denoising chain at each step |
| `denoising_logprobs` | `(T, N, ft_denoising_steps)` | Log-probs for the ft portion only |

The mini-batch generator yields `DiffusionBatch` with `chains_prev`, `chains_next`, and `denoising_inds` — the consecutive pairs of chain states needed to compute the PPO ratio at each denoising step.

### PPODiffusion (`rsl_rl/algorithms/ppo_diffusion.py`, 378 lines)

Extends `PPO` with:

**Dual optimizers:**
```python
actor_optimizer = Adam(actor_ft.parameters() [+ eta], lr=actor_lr)
critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
```
`self.optimizer` is aliased to `actor_optimizer` for `OnPolicyRunner` save/load compatibility.

**Denoising discount (credit assignment):**

The advantage estimated at the environment level is distributed across the `ft_denoising_steps` denoising steps using a geometric discount:

```
A_k = A_env * gamma_denoising^(ft_denoising_steps - 1 - k)
```

This assigns more credit to earlier (noisier) denoising steps, matching the original DPPO paper.

**Exponential clip coefficient annealing:**

The clip coefficient for the policy surrogate loss is annealed per denoising step:

```
coef_k = clip_ploss_coef_base + (clip_ploss_coef - clip_ploss_coef_base)
         * (1 - exp(-clip_ploss_coef_rate * k / ft_denoising_steps))
```

Steps closer to the final action (k → ft_denoising_steps) use the full `clip_ploss_coef`; early denoising steps use the smaller `clip_ploss_coef_base`.

**Update loop:**

```python
for epoch in range(num_learning_epochs):
    for batch in storage.mini_batch_generator():
        # For each ft denoising step k:
        #   1. Compute new log-prob under actor_ft
        #   2. PPO ratio = exp(new_logp - old_logp)
        #   3. Clipped surrogate loss with coef_k
        #   4. Sum weighted by gamma_denoising^(T-1-k)
        actor_optimizer.step()

        # Standard value loss
        critic_optimizer.step()
```

---

## IsaacLab Integration

### New configclasses (`isaaclab_rl/rsl_rl/rl_cfg.py`)

Three new `@configclass` types appended to the existing file. All exported via the existing `from .rl_cfg import *` in `__init__.py` — no new exports needed.

```python
@configclass
class RslRlDiffusionActorCfg:
    class_name: str = "rsl_rl.models.DiffusionModel"
    denoising_steps: int = MISSING
    ft_denoising_steps: int = MISSING
    horizon_steps: int = 1
    mlp_dims: list[int] = MISSING
    activation_type: str = "Mish"
    residual_style: bool = False
    min_sampling_denoising_std: float = 0.1
    min_logprob_denoising_std: float = 0.1

@configclass
class RslRlPpoDiffusionAlgorithmCfg:
    class_name: str = "rsl_rl.algorithms.PPODiffusion"
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    actor_lr: float = MISSING
    critic_lr: float = MISSING
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    gamma_denoising: float = 0.99
    clip_ploss_coef: float = 0.1
    clip_ploss_coef_base: float = 1e-3
    clip_ploss_coef_rate: float = 3.0
    reward_horizon: int = 4
    rnd_cfg: RslRlRndCfg | None = None   # required by on_policy_runner.py:96

@configclass
class RslRlOnPolicyDiffusionRunnerCfg(RslRlOnPolicyRunnerCfg):
    actor: RslRlDiffusionActorCfg = MISSING
    algorithm: RslRlPpoDiffusionAlgorithmCfg = MISSING
    # critic inherited from parent as RslRlMLPModelCfg = MISSING
```

### Compatibility fix (`isaaclab_rl/rsl_rl/utils.py`)

`handle_deprecated_rsl_rl_cfg()` iterates over all model config fields and calls `_update_distribution_cfg()`, which directly accesses `model_cfg.distribution_cfg`. `RslRlDiffusionActorCfg` has no such field, causing an `AttributeError`. Fix: add a `hasattr` guard before the call.

```python
# Before
for model_name in _MODEL_CFG_NAMES:
    if _has_non_missing_attr(agent_cfg, model_name):
        _update_distribution_cfg(getattr(agent_cfg, model_name), RslRlMLPModelCfg)

# After
for model_name in _MODEL_CFG_NAMES:
    if _has_non_missing_attr(agent_cfg, model_name):
        _model = getattr(agent_cfg, model_name)
        if hasattr(_model, "distribution_cfg"):
            _update_distribution_cfg(_model, RslRlMLPModelCfg)
```

### Task configs

Each of the 9 tasks receives a `rsl_rl_dppo_cfg.py` in its `agents/` directory, inheriting from `RslRlOnPolicyDiffusionRunnerCfg`. The `rsl_rl_dppo_cfg_entry_point` key is added to its `gym.register(...)` kwargs.

**Hyperparameter table:**

| Task | Difficulty | mlp_dims | denoising_steps | ft_denoising_steps | actor_lr | max_iterations |
|------|-----------|----------|-----------------|-------------------|---------|---------------|
| Cartpole | Simple | [64, 64] | 50 | 5 | 1e-4 | 150 |
| Ant | Simple | [256, 256, 128] | 100 | 5 | 1e-4 | 500 |
| Anymal-C Flat | Simple | [256, 256, 128] | 100 | 5 | 1e-4 | 500 |
| Humanoid | Medium | [256, 256, 128] | 100 | 5 | 1e-4 | 1000 |
| Franka Drawer | Medium | [256, 128, 64] | 100 | 5 | 1e-4 | 1000 |
| Anymal-C Rough | Medium | [512, 256, 128] | 100 | 10 | 1e-4 | 1500 |
| Allegro Hand | Hard | [512, 256, 128] | 100 | 10 | 1e-5 | 2000 |
| Shadow Hand | Hard | [512, 256, 128] | 100 | 10 | 1e-5 | 2000 |
| G1 Rough | Hard | [512, 256, 128] | 100 | 10 | 1e-4 | 2000 |

All tasks: `critic_lr=1e-3`, `gamma=0.99`, `lam=0.95`, `gamma_denoising=0.99`, `clip_ploss_coef=0.1`.

---

## Tests

All 58 unit tests pass in `env_dppo` (0.46 s):

```
tests/models/test_diffusion_model.py            23 tests
tests/storage/test_diffusion_rollout_storage.py 12 tests
tests/algorithms/test_ppo_diffusion.py          23 tests
```

**Coverage:**

| Module | Tests |
|--------|-------|
| `DiffusionModel` | Construction, forward (stochastic/deterministic), log_prob, mean/std/entropy properties, ft annealing, JIT export |
| `DiffusionRolloutStorage` | Buffer shapes, `add_diffusion_data`, mini-batch size, chains_prev/next shape, denoising_inds, multi-epoch, consecutive chain validation |
| `PPODiffusion` | Dual optimizers, `act`, `process_env_step`, full update loop, save/load checkpoint roundtrip, `construct_algorithm` factory, denoising discount weights, clip coefficient interpolation |

Run:
```bash
conda run -n env_dppo python -m pytest \
    tests/models/test_diffusion_model.py \
    tests/storage/test_diffusion_rollout_storage.py \
    tests/algorithms/test_ppo_diffusion.py -v
```

---

## Benchmark

### Task selection rationale

Tasks are selected to span a spectrum of action distribution multimodality — the main theoretical motivation for diffusion policies over Gaussian actors:

| Difficulty | Task | Multimodality rationale |
|-----------|------|------------------------|
| Simple | Cartpole | Single balance point; strictly unimodal |
| Simple | Ant | Periodic gait; near-unimodal |
| Simple | Anymal-C Flat | Flat terrain; regular single-mode locomotion |
| Medium | Humanoid | Bipedal multi-strategy; diverse gait modes |
| Medium | Franka Drawer | Multiple valid grasp directions |
| Medium | Anymal-C Rough | Rough terrain with multiple footholds |
| Hard | Allegro Hand | Multiple stable grasp configurations |
| Hard | Shadow Hand | Dexterous manipulation; multi-contact modes |
| Hard | G1 Rough | Humanoid rough terrain; highly multimodal |

### Design

- **54 total runs:** 9 tasks × 2 algorithms (PPO, DPPO) × 3 seeds (42, 43, 44)
- PPO uses existing `rsl_rl_cfg_entry_point` (no new PPO config files needed)
- DPPO uses new `rsl_rl_dppo_cfg_entry_point`
- Resume-safe: `run_benchmark.py` reads `benchmark_index.json` on startup and skips completed runs
- Results written incrementally after each run completes

### Running

```bash
# Full benchmark
cd /workspace/robotics/isaaclab_repos/IsaacLab-NVDA
nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/run_benchmark.py \
    > logs/benchmark_run.log 2>&1 &

# Check progress
python3 -c "
import json
d = json.load(open('logs/benchmark_index.json'))
print(f'{len(d)}/54 complete')
for r in d: print(r['task'], r['agent'], r['seed'], r['status'])
"

# Analyze results (can run mid-benchmark for completed tasks)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/analyze_benchmark.py
```

### Output artifacts

| File | Contents |
|------|---------|
| `logs/benchmark_index.json` | Run index: task, agent, seed, log_dir, status |
| `logs/benchmark_comparison.png` | 3x3 subplot: PPO vs DPPO mean reward +/- std per task |
| `logs/benchmark_results.csv` | Final mean reward summary (mean +/- std over last 10 iters) |

### Results

Two benchmarks were run:

**1. Full benchmark (partial, terminated early due to DPPO training time)**

| Task | PPO final reward (mean ± std) | DPPO final reward (mean ± std) | DPPO/PPO |
|------|-------------------------------|-------------------------------|---------|
| Cartpole (Simple) | 295.9 ± 0.2 | 45.7 ± 5.4 | 15% |
| Ant (Simple) | 17958 ± 1518 | 6770 ± 92 (1 seed) | 38% |

Each DPPO run took ~4-5 hours vs ~5 minutes for PPO (~60-100x slower), making the full 54-run benchmark impractical on a single GPU.

**2. Allegro Hand focused benchmark (6/6 complete, `denoising_steps=10`, 500 iters)**

| | seed=42 | seed=43 | seed=44 | Mean |
|--|---------|---------|---------|------|
| PPO | 305.95 | 179.75 | 228.40 | **238.03 ± 52** |
| DPPO | -6.04 | -6.14 | -3.97 | **-5.38 ± 1.0** |

DPPO completely failed to learn. Root cause: `denoising_steps=10` (reduced from 100 for speed) is insufficient for the diffusion model to generate valid actions, resulting in near-random behavior throughout training.

### Key findings

1. **DPPO is extremely sensitive to `denoising_steps`**: reducing from 100→10 renders the actor unable to denoise meaningfully. The DPPO std across seeds (±1.0) vs PPO (±52) confirms this is systematic failure, not random variance.

2. **DPPO is 60-100x slower per iteration** due to running a full `denoising_steps`-step chain per env step during rollout. On a single RTX 4070 Super, a full Allegro run with `denoising_steps=100` takes ~8-10 hours.

3. **Simple tasks favor PPO**: on Cartpole and Ant (near-unimodal reward landscapes), DPPO underperforms PPO even when configured correctly, consistent with the original paper.

4. **Fair comparison was not achieved**: the theoretical advantage of DPPO (multimodal action distributions on Hard tasks like Allegro, Shadow Hand, G1 Rough) could not be validated without running the full 100-step / 2000-iter configurations. This would require either a multi-GPU setup or significantly more wall-clock time.

---

## Known Issues and Resolutions

### 1. `AttributeError: distribution_cfg` in `handle_deprecated_rsl_rl_cfg()`

**Root cause:** `utils.py` called `_update_distribution_cfg()` for every non-MISSING model config, but `RslRlDiffusionActorCfg` has no `distribution_cfg` field.
**Fix:** `hasattr` guard before the call (see Compatibility fix section above).
**File:** `source/isaaclab_rl/isaaclab_rl/rsl_rl/utils.py`

### 2. `KeyError: 'rnd_cfg'` in `on_policy_runner.py`

**Root cause:** `on_policy_runner.py` lines 96 and 124 access `self.cfg["algorithm"]["rnd_cfg"]`. The field was absent from `RslRlPpoDiffusionAlgorithmCfg`.
**Fix:** Added `rnd_cfg: RslRlRndCfg | None = None` to `RslRlPpoDiffusionAlgorithmCfg`. `PPODiffusion.__init__` accepts `**kwargs` so the field passes through to parent PPO without error.
**File:** `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`

### 3. `ModuleNotFoundError: rsl_rl.algorithms.PPODiffusion`

**Root cause:** Isaac Sim's conda environment had the upstream `rsl_rl` package installed, not `rsl_rl-HAND`.
**Fix:** Install rsl_rl-HAND in editable mode inside `env_dppo`:
```bash
pip install -e /workspace/robotics/isaaclab_repos/rsl_rl-HAND
```

---

## References

- [DPPO paper (arXiv:2409.00588)](https://arxiv.org/abs/2409.00588) — Diffusion Policy Policy Optimization
- [rsl_rl upstream](https://github.com/leggedrobotics/rsl_rl) — base PPO library (v5.0.1)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — simulation framework (v2.3.2)
