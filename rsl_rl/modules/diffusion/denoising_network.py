"""
MLP models for diffusion policies.

Ported from dppo/model/diffusion/mlp_diffusion.py (DiffusionMLP only).

"""

import torch
import torch.nn as nn
import logging

from .mlp import MLP, ResidualMLP
from .sinusoidal_emb import SinusoidalPosEmb

log = logging.getLogger(__name__)


class DiffusionMLP(nn.Module):

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + action_dim * horizon_steps + cond_dim
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)

        Also accepts flat tensor cond: (B, Do), which is wrapped as
        {"state": cond.unsqueeze(1)} for compatibility with the original interface.
        """
        # Wrap flat tensor cond into dict format
        if isinstance(cond, torch.Tensor):
            cond = {"state": cond.unsqueeze(1)}  # (B, Do) -> (B, 1, Do)

        B, Ta, Da = x.shape

        # flatten chunk
        x = x.view(B, -1)

        # flatten history
        state = cond["state"].view(B, -1)

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, state], dim=-1)

        # mlp head
        out = self.mlp_mean(x)
        return out.view(B, Ta, Da)
