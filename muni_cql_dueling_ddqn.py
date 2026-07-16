#!/usr/bin/env python3
"""First-version offline CQL-regularized Dueling Double DQN trainer.

Designed for a municipal-bond offer-pricing policy with a discrete action grid:
    action = (price_offset, inventory_fraction), plus one no-quote action.

The trainer consumes offline transitions from either an NPZ file or a directory of memory-mapped NPY arrays. It does not fabricate
counterfactual rewards; each training row must represent a logged or explicitly
simulated transition prepared upstream.

Required transition arrays
--------------------------
states:             float32 [N, state_dim]
actions:            int64   [N]
rewards:            float32 [N]
next_states:        float32 [N, state_dim]
dones:              float32/bool [N]
action_masks:       bool    [N, num_actions]
next_action_masks:  bool    [N, num_actions]

Optional transition arrays
--------------------------
discounts:          float32 [N]
    Per-transition discount. Useful for event-driven steps with irregular time:
    gamma_t = gamma_base ** (delta_minutes / base_minutes).

For large datasets, prefer a directory containing one `<array_name>.npy` file per array.
Run `python muni_cql_dueling_ddqn.py --help` for commands.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger("muni_cql")


# ---------------------------------------------------------------------------
# Reproducibility and utility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp_path)
    os.replace(tmp_path, path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


# ---------------------------------------------------------------------------
# Discrete muni quoting action grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    price_offset: Optional[float]
    quantity_fraction: float
    is_no_quote: bool = False


@dataclass(frozen=True)
class ActionGrid:
    """Maps action IDs to price offsets and inventory fractions."""

    price_offsets: Tuple[float, ...] = (-0.50, -0.25, -0.125, 0.0, 0.125, 0.25, 0.50)
    quantity_fractions: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
    include_no_quote: bool = True

    @property
    def num_actions(self) -> int:
        return len(self.price_offsets) * len(self.quantity_fractions) + int(self.include_no_quote)

    def decode(self, action_id: int) -> ActionSpec:
        if action_id < 0 or action_id >= self.num_actions:
            raise ValueError(f"action_id={action_id} outside [0, {self.num_actions})")
        grid_size = len(self.price_offsets) * len(self.quantity_fractions)
        if self.include_no_quote and action_id == grid_size:
            return ActionSpec(action_id, None, 0.0, True)
        price_index = action_id // len(self.quantity_fractions)
        quantity_index = action_id % len(self.quantity_fractions)
        return ActionSpec(
            action_id=action_id,
            price_offset=self.price_offsets[price_index],
            quantity_fraction=self.quantity_fractions[quantity_index],
            is_no_quote=False,
        )

    def encode(self, price_offset_index: int, quantity_fraction_index: int) -> int:
        if not (0 <= price_offset_index < len(self.price_offsets)):
            raise ValueError("Invalid price_offset_index")
        if not (0 <= quantity_fraction_index < len(self.quantity_fractions)):
            raise ValueError("Invalid quantity_fraction_index")
        return price_offset_index * len(self.quantity_fractions) + quantity_fraction_index

    def no_quote_action_id(self) -> Optional[int]:
        if not self.include_no_quote:
            return None
        return len(self.price_offsets) * len(self.quantity_fractions)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "price_offsets": list(self.price_offsets),
            "quantity_fractions": list(self.quantity_fractions),
            "include_no_quote": self.include_no_quote,
            "num_actions": self.num_actions,
        }


# ---------------------------------------------------------------------------
# Offline transition data
# ---------------------------------------------------------------------------


REQUIRED_ARRAYS = (
    "states",
    "actions",
    "rewards",
    "next_states",
    "dones",
    "action_masks",
    "next_action_masks",
)


class TransitionArrayStore:
    """Load transition arrays from an NPZ file or an NPY-array directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._npz: Optional[Any] = None
        self._arrays: Dict[str, np.ndarray] = {}

        if self.path.is_dir():
            for name in (*REQUIRED_ARRAYS, "discounts"):
                array_path = self.path / f"{name}.npy"
                if array_path.exists():
                    self._arrays[name] = np.load(array_path, mmap_mode="r", allow_pickle=False)
        else:
            self._npz = np.load(self.path, mmap_mode="r", allow_pickle=False)

    def __contains__(self, name: str) -> bool:
        if self._npz is not None:
            return name in self._npz
        return name in self._arrays

    def __getitem__(self, name: str) -> np.ndarray:
        if self._npz is not None:
            return self._npz[name]
        return self._arrays[name]


@dataclass
class StateNormalizer:
    mean: np.ndarray
    std: np.ndarray
    clip: float = 10.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = (x - self.mean) / self.std
        if self.clip > 0:
            y = np.clip(y, -self.clip, self.clip)
        return y.astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": self.std.astype(np.float32).tolist(),
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateNormalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            clip=float(payload.get("clip", 10.0)),
        )


class OfflineTransitionDataset(Dataset[Tuple[Tensor, ...]]):
    """Memory-mapped NPZ offline transition dataset."""

    def __init__(
        self,
        npz_path: str | Path,
        normalizer: StateNormalizer,
        reward_scale: float = 1.0,
        reward_clip: float = 0.0,
        default_discount: float = 0.99,
    ) -> None:
        self.path = Path(npz_path)
        self.data = TransitionArrayStore(self.path)
        missing = [name for name in REQUIRED_ARRAYS if name not in self.data]
        if missing:
            raise ValueError(f"{self.path} is missing arrays: {missing}")

        self.states = self.data["states"]
        self.actions = self.data["actions"]
        self.rewards = self.data["rewards"]
        self.next_states = self.data["next_states"]
        self.dones = self.data["dones"]
        self.action_masks = self.data["action_masks"]
        self.next_action_masks = self.data["next_action_masks"]
        self.discounts = self.data["discounts"] if "discounts" in self.data else None

        self.normalizer = normalizer
        self.reward_scale = float(reward_scale)
        self.reward_clip = float(reward_clip)
        self.default_discount = float(default_discount)
        self._validate_shapes()

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def num_actions(self) -> int:
        return int(self.action_masks.shape[1])

    def _validate_shapes(self) -> None:
        n = len(self.states)
        if self.states.ndim != 2 or self.next_states.shape != self.states.shape:
            raise ValueError("states and next_states must both have shape [N, state_dim]")
        for name, array in (
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("dones", self.dones),
        ):
            if len(array) != n:
                raise ValueError(f"{name} length {len(array)} does not match states length {n}")
        if self.action_masks.ndim != 2 or self.next_action_masks.shape != self.action_masks.shape:
            raise ValueError("action_masks and next_action_masks must have shape [N, num_actions]")
        if len(self.action_masks) != n:
            raise ValueError("action_masks row count must match states")
        if self.discounts is not None and len(self.discounts) != n:
            raise ValueError("discounts length must match states")
        if self.normalizer.mean.shape != (self.state_dim,) or self.normalizer.std.shape != (self.state_dim,):
            raise ValueError("Normalizer dimension does not match state_dim")

        actions = np.asarray(self.actions)
        if actions.min(initial=0) < 0 or actions.max(initial=0) >= self.num_actions:
            raise ValueError("actions contain IDs outside action-mask width")

        # Sample rows instead of materializing the full masks for very large files.
        sample_size = min(n, 50_000)
        if sample_size:
            sample_idx = np.linspace(0, n - 1, sample_size, dtype=np.int64)
            sampled_actions = np.asarray(self.actions[sample_idx], dtype=np.int64)
            sampled_masks = np.asarray(self.action_masks[sample_idx], dtype=bool)
            if not np.all(sampled_masks[np.arange(sample_size), sampled_actions]):
                raise ValueError("At least one sampled logged action is invalid under action_masks")
            sampled_next_masks = np.asarray(self.next_action_masks[sample_idx], dtype=bool)
            sampled_dones = np.asarray(self.dones[sample_idx], dtype=bool)
            no_valid_next = sampled_next_masks.sum(axis=1) == 0
            if np.any(no_valid_next & ~sampled_dones):
                raise ValueError("A nonterminal sampled row has no valid next action")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        state = self.normalizer.transform(np.asarray(self.states[idx], dtype=np.float32))
        next_state = self.normalizer.transform(np.asarray(self.next_states[idx], dtype=np.float32))
        reward = float(self.rewards[idx]) / self.reward_scale
        if self.reward_clip > 0:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        discount = float(self.discounts[idx]) if self.discounts is not None else self.default_discount

        return (
            torch.from_numpy(state),
            torch.tensor(int(self.actions[idx]), dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.from_numpy(next_state),
            torch.tensor(float(self.dones[idx]), dtype=torch.float32),
            torch.from_numpy(np.array(self.action_masks[idx], dtype=bool, copy=True)),
            torch.from_numpy(np.array(self.next_action_masks[idx], dtype=bool, copy=True)),
            torch.tensor(discount, dtype=torch.float32),
        )


def compute_state_normalizer(
    npz_path: str | Path,
    chunk_size: int = 200_000,
    min_std: float = 1e-6,
    clip: float = 10.0,
) -> StateNormalizer:
    """Compute state mean/std from training states only, in chunks."""
    data = TransitionArrayStore(npz_path)
    if "states" not in data:
        raise ValueError(f"{npz_path} does not contain 'states'")
    states = data["states"]
    if states.ndim != 2:
        raise ValueError("states must be [N, state_dim]")

    n, dim = states.shape
    count = 0
    mean = np.zeros(dim, dtype=np.float64)
    m2 = np.zeros(dim, dtype=np.float64)

    for start in range(0, n, chunk_size):
        chunk = np.asarray(states[start : start + chunk_size], dtype=np.float64)
        if not np.isfinite(chunk).all():
            bad = int(np.size(chunk) - np.isfinite(chunk).sum())
            raise ValueError(f"states contain {bad} non-finite values in chunk starting at {start}")
        chunk_count = len(chunk)
        chunk_mean = chunk.mean(axis=0)
        chunk_m2 = ((chunk - chunk_mean) ** 2).sum(axis=0)

        if count == 0:
            mean = chunk_mean
            m2 = chunk_m2
            count = chunk_count
            continue

        delta = chunk_mean - mean
        total = count + chunk_count
        mean += delta * (chunk_count / total)
        m2 += chunk_m2 + delta**2 * count * chunk_count / total
        count = total

    variance = m2 / max(count - 1, 1)
    std = np.sqrt(np.maximum(variance, min_std**2))
    return StateNormalizer(mean.astype(np.float32), std.astype(np.float32), clip=clip)


# ---------------------------------------------------------------------------
# Dueling Q-network
# ---------------------------------------------------------------------------


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
            ]
        )
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class DuelingQNetwork(nn.Module):
    """Dueling Q-network with a valid-action-aware advantage centering term."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        trunk_dims: Sequence[int] = (512, 256),
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        if not trunk_dims:
            raise ValueError("trunk_dims cannot be empty")
        trunk_layers: List[nn.Module] = []
        previous = state_dim
        for hidden in trunk_dims:
            trunk_layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            previous = hidden
        self.trunk = nn.Sequential(*trunk_layers)
        self.value_head = build_mlp(previous, (head_dim,), 1)
        self.advantage_head = build_mlp(previous, (head_dim,), num_actions)
        self.num_actions = num_actions
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        # Keep initial Q-values near zero.
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.advantage_head[-1].weight, gain=0.01)

    def forward(self, state: Tensor, action_mask: Optional[Tensor] = None) -> Tensor:
        features = self.trunk(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)

        if action_mask is None:
            centered_advantage = advantage - advantage.mean(dim=1, keepdim=True)
        else:
            mask = action_mask.to(dtype=advantage.dtype)
            valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            valid_mean = (advantage * mask).sum(dim=1, keepdim=True) / valid_count
            centered_advantage = advantage - valid_mean
        return value + centered_advantage


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    seed: int = 2026
    device: str = "auto"
    epochs: int = 30
    batch_size: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gamma: float = 0.99
    reward_scale: float = 1.0
    reward_clip: float = 10.0
    cql_alpha: float = 1.0
    cql_temperature: float = 1.0
    target_update_interval: int = 1_000
    gradient_clip_norm: float = 10.0
    num_workers: int = 0
    pin_memory: bool = True
    trunk_dims: Tuple[int, ...] = (512, 256)
    head_dim: int = 128
    normalizer_clip: float = 10.0
    log_interval: int = 100
    patience: int = 8
    min_delta: float = 1e-4
    max_train_batches_per_epoch: int = 0
    max_valid_batches: int = 0
    torch_num_threads: int = 8


@dataclass
class EpochMetrics:
    total_loss: float
    td_loss: float
    cql_loss: float
    cql_gap: float
    data_q: float
    max_valid_q: float
    behavior_agreement: float
    samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {
            "total_loss": 0.0,
            "td_loss": 0.0,
            "cql_loss": 0.0,
            "cql_gap": 0.0,
            "data_q": 0.0,
            "max_valid_q": 0.0,
            "behavior_agreement": 0.0,
        }
        self.samples = 0

    def update(self, metrics: Mapping[str, float], batch_size: int) -> None:
        for key in self.sums:
            self.sums[key] += float(metrics[key]) * batch_size
        self.samples += batch_size

    def finalize(self) -> EpochMetrics:
        denominator = max(self.samples, 1)
        return EpochMetrics(
            **{key: value / denominator for key, value in self.sums.items()},
            samples=self.samples,
        )


class CQLDuelingDoubleDQNTrainer:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.online = DuelingQNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            trunk_dims=config.trunk_dims,
            head_dim=config.head_dim,
        ).to(device)
        self.target = DuelingQNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            trunk_dims=config.trunk_dims,
            head_dim=config.head_dim,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            self.online.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.global_step = 0
        self.num_actions = num_actions

    @staticmethod
    def _mask_q(q_values: Tensor, action_mask: Tensor) -> Tensor:
        if action_mask.dtype is not torch.bool:
            action_mask = action_mask.bool()
        return q_values.masked_fill(~action_mask, torch.finfo(q_values.dtype).min)

    def _compute_loss(
        self,
        batch: Tuple[Tensor, ...],
        cql_alpha: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            action_masks,
            next_action_masks,
            discounts,
        ) = (tensor.to(self.device, non_blocking=True) for tensor in batch)

        q_all = self.online(states, action_masks)
        q_data = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online(next_states, next_action_masks)
            next_q_online_masked = self._mask_q(next_q_online, next_action_masks)
            next_actions = next_q_online_masked.argmax(dim=1)

            next_q_target = self.target(next_states, next_action_masks)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + discounts * (1.0 - dones) * next_q

        td_loss = nn.functional.huber_loss(q_data, td_target, delta=1.0)

        # Discrete-action CQL(H): logsumexp over valid actions minus logged-action Q.
        temperature = self.config.cql_temperature
        q_valid = self._mask_q(q_all, action_masks)
        conservative_value = temperature * torch.logsumexp(q_valid / temperature, dim=1)
        cql_gap_per_row = conservative_value - q_data
        cql_gap = cql_gap_per_row.mean()
        effective_cql_alpha = self.config.cql_alpha if cql_alpha is None else float(cql_alpha)
        cql_loss = effective_cql_alpha * cql_gap
        total_loss = td_loss + cql_loss

        with torch.no_grad():
            greedy_actions = q_valid.argmax(dim=1)
            behavior_agreement = (greedy_actions == actions).float().mean()
            max_valid_q = q_valid.max(dim=1).values.mean()

        metrics = {
            "total_loss": float(total_loss.detach().cpu()),
            "td_loss": float(td_loss.detach().cpu()),
            "cql_loss": float(cql_loss.detach().cpu()),
            "cql_gap": float(cql_gap.detach().cpu()),
            "data_q": float(q_data.mean().detach().cpu()),
            "max_valid_q": float(max_valid_q.detach().cpu()),
            "behavior_agreement": float(behavior_agreement.detach().cpu()),
            "cql_alpha": float(effective_cql_alpha),
        }
        return total_loss, metrics

    def update_batch(
        self,
        batch: Tuple[Tensor, ...],
        cql_alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """Run one optimizer step on an offline or simulator replay batch."""
        self.online.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self._compute_loss(batch, cql_alpha=cql_alpha)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at global_step={self.global_step}: {metrics}"
            )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.global_step += 1
        if self.global_step % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        metrics = dict(metrics)
        metrics["gradient_norm"] = float(grad_norm.detach().cpu())
        metrics["global_step"] = float(self.global_step)
        return metrics

    def copy_online_to_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def train_epoch(self, loader: DataLoader[Tuple[Tensor, ...]], epoch: int) -> EpochMetrics:
        self.online.train()
        accumulator = MetricAccumulator()
        start_time = time.time()

        for batch_index, batch in enumerate(loader, start=1):
            if self.config.max_train_batches_per_epoch and batch_index > self.config.max_train_batches_per_epoch:
                break
            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at global_step={self.global_step}: {metrics}")
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()

            self.global_step += 1
            if self.global_step % self.config.target_update_interval == 0:
                self.target.load_state_dict(self.online.state_dict())

            batch_size = int(batch[0].shape[0])
            accumulator.update(metrics, batch_size)

            if self.config.log_interval and batch_index % self.config.log_interval == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                LOGGER.info(
                    "epoch=%d batch=%d step=%d loss=%.5f td=%.5f cql=%.5f "
                    "agree=%.3f rows/s=%.0f",
                    epoch,
                    batch_index,
                    self.global_step,
                    metrics["total_loss"],
                    metrics["td_loss"],
                    metrics["cql_loss"],
                    metrics["behavior_agreement"],
                    accumulator.samples / elapsed,
                )

        return accumulator.finalize()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[Tuple[Tensor, ...]]) -> EpochMetrics:
        self.online.eval()
        accumulator = MetricAccumulator()
        for batch_index, batch in enumerate(loader, start=1):
            if self.config.max_valid_batches and batch_index > self.config.max_valid_batches:
                break
            _, metrics = self._compute_loss(batch)
            accumulator.update(metrics, int(batch[0].shape[0]))
        return accumulator.finalize()

    @torch.no_grad()
    def action_distribution(self, loader: DataLoader[Tuple[Tensor, ...]]) -> Dict[str, Any]:
        self.online.eval()
        counts = np.zeros(self.num_actions, dtype=np.int64)
        behavior_counts = np.zeros(self.num_actions, dtype=np.int64)
        total = 0
        for batch_index, batch in enumerate(loader, start=1):
            if self.config.max_valid_batches and batch_index > self.config.max_valid_batches:
                break
            states, actions, _, _, _, masks, _, _ = batch
            states = states.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            q_values = self._mask_q(self.online(states, masks), masks)
            selected = q_values.argmax(dim=1).cpu().numpy()
            logged = actions.numpy()
            counts += np.bincount(selected, minlength=self.num_actions)
            behavior_counts += np.bincount(logged, minlength=self.num_actions)
            total += len(selected)
        return {
            "total": int(total),
            "policy_counts": counts.tolist(),
            "behavior_counts": behavior_counts.tolist(),
            "policy_fractions": (counts / max(total, 1)).tolist(),
            "behavior_fractions": (behavior_counts / max(total, 1)).tolist(),
        }

    @torch.no_grad()
    def select_action(self, normalized_state: np.ndarray, action_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        self.online.eval()
        state_tensor = torch.as_tensor(normalized_state, dtype=torch.float32, device=self.device).view(1, -1)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).view(1, -1)
        q_values = self._mask_q(self.online(state_tensor, mask_tensor), mask_tensor)
        action_id = int(q_values.argmax(dim=1).item())
        return action_id, q_values.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def create_loader(
    dataset: OfflineTransitionDataset,
    config: TrainConfig,
    shuffle: bool,
) -> DataLoader[Tuple[Tensor, ...]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=config.num_workers > 0,
    )


def save_training_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def run_training(args: argparse.Namespace) -> None:
    config = TrainConfig(
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        cql_alpha=args.cql_alpha,
        cql_temperature=args.cql_temperature,
        target_update_interval=args.target_update_interval,
        gradient_clip_norm=args.gradient_clip_norm,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        trunk_dims=tuple(args.trunk_dims),
        head_dim=args.head_dim,
        normalizer_clip=args.normalizer_clip,
        log_interval=args.log_interval,
        patience=args.patience,
        min_delta=args.min_delta,
        max_train_batches_per_epoch=args.max_train_batches_per_epoch,
        max_valid_batches=args.max_valid_batches,
        torch_num_threads=args.torch_num_threads,
    )
    if config.torch_num_threads > 0:
        torch.set_num_threads(config.torch_num_threads)
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("device=%s", device)
    LOGGER.info("computing state normalizer from training states only")
    normalizer = compute_state_normalizer(args.train_npz, clip=config.normalizer_clip)

    train_dataset = OfflineTransitionDataset(
        args.train_npz,
        normalizer,
        reward_scale=config.reward_scale,
        reward_clip=config.reward_clip,
        default_discount=config.gamma,
    )
    valid_dataset = OfflineTransitionDataset(
        args.valid_npz,
        normalizer,
        reward_scale=config.reward_scale,
        reward_clip=config.reward_clip,
        default_discount=config.gamma,
    )
    if train_dataset.state_dim != valid_dataset.state_dim:
        raise ValueError("train and validation state dimensions differ")
    if train_dataset.num_actions != valid_dataset.num_actions:
        raise ValueError("train and validation action counts differ")

    action_grid = ActionGrid()
    if train_dataset.num_actions != action_grid.num_actions:
        LOGGER.warning(
            "dataset has %d actions, while default ActionGrid has %d. "
            "Training will continue, but update ActionGrid metadata for production inference.",
            train_dataset.num_actions,
            action_grid.num_actions,
        )

    train_loader = create_loader(train_dataset, config, shuffle=True)
    valid_loader = create_loader(valid_dataset, config, shuffle=False)
    trainer = CQLDuelingDoubleDQNTrainer(
        state_dim=train_dataset.state_dim,
        num_actions=train_dataset.num_actions,
        config=config,
        device=device,
    )

    history: List[Dict[str, Any]] = []
    best_score = float("inf")
    stale_epochs = 0
    best_path = output_dir / "best_checkpoint.pt"
    last_path = output_dir / "last_checkpoint.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        valid_metrics = trainer.evaluate(valid_loader)
        row = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "train": train_metrics.to_dict(),
            "valid": valid_metrics.to_dict(),
        }
        history.append(row)
        LOGGER.info(
            "epoch=%d train_loss=%.5f valid_loss=%.5f valid_td=%.5f "
            "valid_cql=%.5f valid_agreement=%.3f",
            epoch,
            train_metrics.total_loss,
            valid_metrics.total_loss,
            valid_metrics.td_loss,
            valid_metrics.cql_loss,
            valid_metrics.behavior_agreement,
        )

        checkpoint = {
            "model_state_dict": trainer.online.state_dict(),
            "target_state_dict": trainer.target.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "global_step": trainer.global_step,
            "epoch": epoch,
            "state_dim": train_dataset.state_dim,
            "num_actions": train_dataset.num_actions,
            "config": asdict(config),
            "normalizer": normalizer.to_dict(),
            "action_grid": action_grid.to_json_dict(),
            "train_metrics": train_metrics.to_dict(),
            "valid_metrics": valid_metrics.to_dict(),
        }
        atomic_torch_save(checkpoint, last_path)

        # Validation total loss is only a training diagnostic. Final policy selection
        # should also use a historical replay simulator or fitted-Q evaluation.
        score = valid_metrics.total_loss
        if score < best_score - config.min_delta:
            best_score = score
            stale_epochs = 0
            atomic_torch_save(checkpoint, best_path)
            LOGGER.info("saved new best checkpoint: %s", best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                LOGGER.info("early stopping after %d stale epochs", stale_epochs)
                break

        save_training_summary(output_dir / "history.json", {"history": history})

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    trainer.online.load_state_dict(best_checkpoint["model_state_dict"])
    distribution = trainer.action_distribution(valid_loader)
    summary = {
        "best_validation_total_loss": best_score,
        "best_epoch": int(best_checkpoint["epoch"]),
        "state_dim": train_dataset.state_dim,
        "num_actions": train_dataset.num_actions,
        "config": asdict(config),
        "normalizer": normalizer.to_dict(),
        "action_grid": action_grid.to_json_dict(),
        "validation_action_distribution": distribution,
        "history": history,
        "important_note": (
            "Validation TD/CQL loss is not an unbiased estimate of policy value. "
            "Use chronological historical replay and/or fitted-Q evaluation before promotion."
        ),
    }
    save_training_summary(output_dir / "training_summary.json", summary)
    LOGGER.info("training complete; artifacts written to %s", output_dir)


def load_policy(checkpoint_path: str | Path, device: torch.device) -> Tuple[CQLDuelingDoubleDQNTrainer, StateNormalizer, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_payload = dict(checkpoint["config"])
    config_payload["trunk_dims"] = tuple(config_payload["trunk_dims"])
    config = TrainConfig(**config_payload)
    trainer = CQLDuelingDoubleDQNTrainer(
        state_dim=int(checkpoint["state_dim"]),
        num_actions=int(checkpoint["num_actions"]),
        config=config,
        device=device,
    )
    trainer.online.load_state_dict(checkpoint["model_state_dict"])
    trainer.target.load_state_dict(checkpoint["target_state_dict"])
    normalizer = StateNormalizer.from_dict(checkpoint["normalizer"])
    return trainer, normalizer, checkpoint


def run_inference(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    trainer, normalizer, checkpoint = load_policy(args.checkpoint, device)
    state = np.load(args.state_npy, allow_pickle=False).astype(np.float32)
    mask = np.load(args.mask_npy, allow_pickle=False).astype(bool)
    if state.ndim != 1:
        raise ValueError("state_npy must contain one state vector with shape [state_dim]")
    if mask.ndim != 1:
        raise ValueError("mask_npy must contain one mask vector with shape [num_actions]")
    normalized = normalizer.transform(state)
    action_id, q_values = trainer.select_action(normalized, mask)

    grid_payload = checkpoint.get("action_grid", {})
    grid = ActionGrid(
        price_offsets=tuple(grid_payload.get("price_offsets", ActionGrid().price_offsets)),
        quantity_fractions=tuple(grid_payload.get("quantity_fractions", ActionGrid().quantity_fractions)),
        include_no_quote=bool(grid_payload.get("include_no_quote", True)),
    )
    action_spec = grid.decode(action_id) if grid.num_actions == len(mask) else None
    result = {
        "action_id": action_id,
        "action_spec": dataclasses.asdict(action_spec) if action_spec else None,
        "q_values": q_values.tolist(),
    }
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Synthetic smoke-test data
# ---------------------------------------------------------------------------


def make_demo_dataset(path: Path, n: int, state_dim: int, action_grid: ActionGrid, seed: int) -> None:
    rng = np.random.default_rng(seed)
    num_actions = action_grid.num_actions
    states = rng.normal(size=(n, state_dim)).astype(np.float32)
    next_states = (0.92 * states + 0.15 * rng.normal(size=(n, state_dim))).astype(np.float32)

    masks = np.ones((n, num_actions), dtype=bool)
    next_masks = np.ones((n, num_actions), dtype=bool)
    # Randomly mask a few aggressive actions to exercise action masking.
    random_invalid = rng.random((n, num_actions)) < 0.05
    masks &= ~random_invalid
    next_masks &= ~(rng.random((n, num_actions)) < 0.05)
    masks[:, action_grid.no_quote_action_id()] = True
    next_masks[:, action_grid.no_quote_action_id()] = True

    latent = states[:, : min(4, state_dim)].sum(axis=1)
    preferred_price_bucket = np.clip(((latent - latent.min()) / (np.ptp(latent) + 1e-6) * 6).astype(int), 0, 6)
    preferred_qty_bucket = np.clip((np.abs(states[:, 0]) * 2).astype(int), 0, 4)
    behavior_action = preferred_price_bucket * 5 + preferred_qty_bucket
    exploratory = rng.random(n) < 0.25
    behavior_action[exploratory] = rng.integers(0, num_actions - 1, exploratory.sum())
    # Repair actions that were masked.
    for i in range(n):
        if not masks[i, behavior_action[i]]:
            behavior_action[i] = int(np.flatnonzero(masks[i])[0])

    price_offsets = np.array(
        [action_grid.decode(i).price_offset or 0.0 for i in range(num_actions)], dtype=np.float32
    )
    qty_fractions = np.array(
        [action_grid.decode(i).quantity_fraction for i in range(num_actions)], dtype=np.float32
    )
    chosen_offset = price_offsets[behavior_action]
    chosen_qty = qty_fractions[behavior_action]
    rewards = (
        0.3 * latent
        + 1.2 * chosen_qty
        - 1.8 * chosen_offset**2
        - 0.7 * (chosen_qty - 0.5 - 0.1 * np.tanh(latent)) ** 2
        + rng.normal(scale=0.2, size=n)
    ).astype(np.float32)
    dones = (rng.random(n) < 0.04).astype(np.float32)
    discounts = np.power(0.99, rng.uniform(0.5, 3.0, size=n)).astype(np.float32)

    np.savez_compressed(
        path,
        states=states,
        actions=behavior_action.astype(np.int64),
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        action_masks=masks,
        next_action_masks=next_masks,
        discounts=discounts,
    )


def run_make_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = ActionGrid()
    make_demo_dataset(output_dir / "demo_train.npz", args.train_rows, args.state_dim, grid, args.seed)
    make_demo_dataset(output_dir / "demo_valid.npz", args.valid_rows, args.state_dim, grid, args.seed + 1)
    with (output_dir / "action_grid.json").open("w", encoding="utf-8") as handle:
        json.dump(grid.to_json_dict(), handle, indent=2)
    LOGGER.info("demo datasets written to %s", output_dir)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo", help="Create synthetic NPZ files for a smoke test")
    demo.add_argument("--output-dir", required=True)
    demo.add_argument("--train-rows", type=int, default=50_000)
    demo.add_argument("--valid-rows", type=int, default=10_000)
    demo.add_argument("--state-dim", type=int, default=64)
    demo.add_argument("--seed", type=int, default=2026)
    demo.set_defaults(func=run_make_demo)

    train = subparsers.add_parser("train", help="Train offline CQL Dueling Double DQN")
    train.add_argument("--train-npz", required=True, help="NPZ file or NPY-array directory")
    train.add_argument("--valid-npz", required=True, help="NPZ file or NPY-array directory")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--reward-clip", type=float, default=10.0)
    train.add_argument("--cql-alpha", type=float, default=1.0)
    train.add_argument("--cql-temperature", type=float, default=1.0)
    train.add_argument("--target-update-interval", type=int, default=1000)
    train.add_argument("--gradient-clip-norm", type=float, default=10.0)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--no-pin-memory", action="store_true")
    train.add_argument("--trunk-dims", type=int, nargs="+", default=(512, 256))
    train.add_argument("--head-dim", type=int, default=128)
    train.add_argument("--normalizer-clip", type=float, default=10.0)
    train.add_argument("--log-interval", type=int, default=100)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--min-delta", type=float, default=1e-4)
    train.add_argument("--max-train-batches-per-epoch", type=int, default=0)
    train.add_argument("--max-valid-batches", type=int, default=0)
    train.add_argument("--torch-num-threads", type=int, default=8)
    train.set_defaults(func=run_training)

    infer = subparsers.add_parser("infer", help="Select one action from a saved checkpoint")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--state-npy", required=True)
    infer.add_argument("--mask-npy", required=True)
    infer.add_argument("--device", default="auto")
    infer.set_defaults(func=run_inference)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
