"""Continuous flow matching generator for particle phase space.

A velocity network is trained with the straight-line flow matching objective
to transport standard normal noise to the training data. Sampling integrates
the learned velocity field from ``t = 0`` to ``t = 1`` with explicit Euler
steps. Features are mapped to a Gaussian space with a per-feature
Gauss rank transform before training and mapped back after sampling.

Requires PyTorch (``pip install aisource-neutrons[torch]``).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseGenerator, Particle
from .preprocessing import GaussRankTransform


class ResBlock(nn.Module):
    """Residual MLP block modulated by a conditioning embedding."""

    def __init__(self, width: int, emb_dim: int) -> None:
        super().__init__()
        self.emb = nn.Linear(emb_dim, 2 * width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        scale, shift = self.emb(e).chunk(2, dim=-1)
        y = x * (1.0 + scale) + shift
        y = self.fc2(F.silu(self.fc1(y)))
        return x + y


class VelocityField(nn.Module):
    """Network predicting the flow velocity ``v(x, t)``."""

    def __init__(self, input_dim: int = 12, width: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, width),
        )
        self.in_proj = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width, width) for _ in range(depth)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, input_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        e = self.time_mlp(t)
        x = self.in_proj(x) + e
        for block in self.blocks:
            x = block(x, e)
        return self.out(x)


class Sampler(nn.Module):
    """Integrate a velocity field from noise to data with Euler steps."""

    def __init__(self, velocity_model: nn.Module, n_steps: int = 64) -> None:
        super().__init__()
        self.velocity = velocity_model
        self.n_steps = n_steps
        self.register_buffer("time_table", torch.linspace(0, 1, n_steps).view(n_steps, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            x = x + dt * self.velocity(x, self.time_table[step])
        return x


def flow_matching_loss(
    model: nn.Module, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Mean squared error to the straight-line velocity ``x1 - x0``."""

    xt = (1 - t) * x0 + t * x1
    v_pred = model(xt, t)
    return ((v_pred - (x1 - x0)) ** 2).mean(dim=-1).mean()


def sample_time(
    batch_size: int, device: torch.device | str
) -> torch.Tensor:
    """Draw training times: 70% uniform, 20% near one, 10% near zero."""

    def uniform() -> torch.Tensor:
        return torch.rand(batch_size, 1, device=device)

    mode = uniform()
    t_near_1 = 1.0 - uniform().pow(2)
    t_near_0 = uniform().pow(2)
    return torch.where(mode < 0.70, uniform(), torch.where(mode < 0.90, t_near_1, t_near_0))


class ContinuousFlowMatching(BaseGenerator):
    """Flow matching generator with an exponential-moving-average velocity network.

    ``fit`` trains for ``steps`` mini-batches, tracks the validation loss of the
    EMA network every ``eval_every`` steps, and keeps the best EMA weights.
    Training and validation losses are stored in ``losses`` and ``val_losses``.

    Without a ``validation`` array, ``val_size`` rows (at most a tenth of the
    training rows) are held out from ``train``.
    """

    def __init__(
        self,
        particle: Particle,
        *,
        width: int = 64,
        depth: int = 3,
        n_steps: int = 64,
        steps: int = 10_000,
        lr: float = 1e-3,
        batch_size: int = 3_000,
        val_size: int = 100_000,
        ema_decay: float = 0.999,
        eval_every: int = 1_000,
        gauss_rank: bool = True,
        device: str = "cpu",
        seed: int = 17,
    ) -> None:
        super().__init__(particle)
        self.width = width
        self.depth = depth
        self.n_steps = n_steps
        self.steps = steps
        self.lr = lr
        self.batch_size = batch_size
        self.val_size = val_size
        self.ema_decay = ema_decay
        self.eval_every = eval_every
        self.gauss_rank = gauss_rank
        self.device = device
        self.seed = seed
        self.losses: list[float] = []
        self.val_losses: list[tuple[int, float]] = []

    def fit(self, train: np.ndarray, validation: np.ndarray) -> None:
        train = self._check(train)
        validation = self._check(validation)

        self.transform = GaussRankTransform().fit(train)
        train_x = self._to_tensor(self._forward(train))
        val_x1 = self._to_tensor(self._forward(validation))

        model = VelocityField(self.ndim, self.width, self.depth).to(self.device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        val_x0 = torch.randn(val_x1.shape, device=self.device)
        val_t = torch.rand(len(val_x1), 1, device=self.device)

        self.losses, self.val_losses = [], []
        best_loss, best_state = float("inf"), copy.deepcopy(ema.state_dict())
        for step in range(self.steps):
            index = torch.randint(0, len(train_x), (self.batch_size,), device=self.device)
            x1 = train_x[index]
            x0 = torch.randn(x1.shape, device=self.device)
            t = sample_time(self.batch_size, self.device)
            loss = flow_matching_loss(model, x0, x1, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._update_ema(ema, model)
            self.losses.append(loss.item())

            if step % self.eval_every == 0 or step == self.steps - 1:
                with torch.no_grad():
                    val_loss = flow_matching_loss(ema, val_x0, val_x1, val_t).item()
                self.val_losses.append((step, val_loss))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(ema.state_dict())

        ema.load_state_dict(best_state)
        self.model = ema

    def sample(self, n: int, seed: int) -> np.ndarray:
        sampler = Sampler(self.model, self.n_steps).to(self.device)
        batches = []
        with torch.no_grad():
            for start in range(0, n, 10_000):
                noise = torch.randn(
                    (min(10_000, n - start), self.ndim), device=self.device
                )
                batches.append(sampler(noise).cpu().numpy())
        samples = np.concatenate(batches) if batches else np.empty((0, self.ndim))
        return self.transform.inverse_transform(samples).astype(np.float32)

    def _forward(self, values: np.ndarray) -> np.ndarray:
        return values if self.transform is None else self.transform.transform(values)

    def _update_ema(self, ema: nn.Module, model: nn.Module) -> None:
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters(), strict=True):
                p_ema.mul_(self.ema_decay).add_(p, alpha=1.0 - self.ema_decay)
