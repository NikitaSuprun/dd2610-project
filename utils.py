from dataclasses import dataclass
from enum import Enum
import tomllib

import torch
from torch import nn


class ConditionType(str, Enum):
    T_R = "t_r"
    T_DELTA_T = "t_delta_t"
    # TODO: Potentially add (t, r, t-r) and (t-r) later


class SamplerType(str, Enum):
    UNIFORM = "uniform"
    LOGNORM = "lognorm"


@dataclass
class TrainingConfig:
    mode: str
    model_config: str
    condition_type: ConditionType
    input_size: int
    batch_size: int
    n_steps: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    gamma: float
    c: float
    sample_ratio: float
    sampler_type: str
    sample_params: dict
    log_step: int
    sample_step: int
    checkpoint_step: int
    histogram_step: int
    mixed_precision: str
    jvp_use_autograd: bool
    n_samples_per_class: int
    sampling_steps: int
    sample_grid_nrow: int


@dataclass
class ModelConfig:
    depth: int
    hidden_dim: int
    num_heads: int
    patch_size: int


class Config:
    """Configuration loader for TOML files."""

    @staticmethod
    def from_toml(filepath: str) -> dict:
        """Load configuration from TOML file and return as dict."""
        with open(filepath, "rb") as f:
            data = tomllib.load(f)

        # Parse training config
        training_data = data.get("training", {})
        training_config = TrainingConfig(
            mode=training_data.get("mode", "debug"),
            model_config=training_data.get("model_config", "B2"),
            condition_type=ConditionType(training_data.get("condition_type", "t_r")),
            input_size=training_data.get("input_size", 32),
            batch_size=training_data.get("batch_size", 48),
            n_steps=training_data.get("n_steps", 20000),
            num_workers=training_data.get("num_workers", 8),
            learning_rate=training_data.get("learning_rate", 0.001),
            weight_decay=training_data.get("weight_decay", 0.0001),
            gamma=training_data.get("gamma", 0.5),
            c=training_data.get("c", 1.0),
            sample_ratio=training_data.get("sample_ratio", 0.5),
            sampler_type=training_data.get("sampler_type", "uniform"),
            sample_params=training_data.get("sample_params", {}),
            log_step=training_data.get("log_step", 500),
            sample_step=training_data.get("sample_step", 5000),
            checkpoint_step=training_data.get("checkpoint_step", 5000),
            histogram_step=training_data.get("histogram_step", 2000),
            mixed_precision=training_data.get("mixed_precision", "fp16"),
            jvp_use_autograd=training_data.get("jvp_use_autograd", False),
            n_samples_per_class=training_data.get("n_samples_per_class", 1),
            sampling_steps=training_data.get("sampling_steps", 5),
            sample_grid_nrow=training_data.get("sample_grid_nrow", 10),
        )

        # Parse model configs
        model_configs = {}
        if "model" in data and isinstance(data["model"], dict):
            for model_name, model_data in data["model"].items():
                model_configs[model_name] = ModelConfig(
                    depth=model_data.get("depth", 12),
                    hidden_dim=model_data.get("hidden_dim", 768),
                    num_heads=model_data.get("num_heads", 12),
                    patch_size=model_data.get("patch_size", 2),
                )

        return {
            "training": training_config,
            "model": model_configs,
        }


class TRSampler:
    def __init__(
        self,
        sample_ratio: float,
        sampler_type: SamplerType,
        sample_params: dict[str, float],
    ):
        assert 0.0 <= sample_ratio <= 1.0, "sample_ratio must be in [0, 1]"
        self.sample_ratio = sample_ratio

        if sampler_type == SamplerType.UNIFORM:
            low = float(sample_params.get("low", sample_params.get("min", 0.0)))
            high = float(sample_params.get("high", sample_params.get("max", 1.0)))
            assert high > low, "Uniform sampler requires high > low"
            self.sampler_fn = lambda batch_size: (
                low + (high - low) * torch.rand(batch_size, 2, dtype=torch.float32)
            )
        elif sampler_type == SamplerType.LOGNORM:
            mean = float(sample_params.get("mean", 0.0))
            sigma = float(sample_params.get("sigma", 1.0))
            dist = torch.distributions.LogNormal(mean, sigma)
            self.sampler_fn = lambda batch_size, dist=dist: torch.sigmoid(
                dist.sample((batch_size, 2)).to(torch.float32)
            )
        else:
            raise ValueError(f"Unsupported sampler_type: {sampler_type}")

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples = self.sampler_fn(batch_size).to(device)  # (batch_size, 2)

        # Assign t = max, r = min, for each pair
        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1])

        # For a fraction sample_ratio of the batch, randomly selects and force r = t on those
        num_selected = int(self.sample_ratio * batch_size)
        indices = torch.randperm(batch_size, device=device)[:num_selected]
        r[indices] = t[indices]

        return t, r


class Loss:
    def __init__(self, gamma: float, c: float):
        self.gamma = gamma
        self.c = c

    def __call__(self, error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # error: (B, ...)

        # Per-sample mean squared error over (C ,H, W, ...)
        delta_squared = torch.mean(error**2, dim=tuple(range(1, error.ndim)))  # (B,)
        mse = delta_squared.mean()  # Simple MSE for logging
        weight = 1 / (delta_squared + self.c).pow(1 - self.gamma)  # (B,)
        weighted_loss = (stopgrad(weight) * delta_squared).mean()  # scalar
        return weighted_loss, mse


class JVP:
    def __init__(self, use_autograd: bool, jvp_config: tuple[int, int]):
        self.use_autograd = use_autograd

        # Used for ablation studies
        self.jvp_config = jvp_config  # Ground truth is (1, 0)

    def __call__(self, model, z, t, r, y, v_hat) -> tuple[torch.Tensor, torch.Tensor]:
        model_partial = lambda z, t, r: model(z, t, r, y)
        grad_params = (
            v_hat,
            torch.ones_like(t) * self.jvp_config[0],
            torch.ones_like(r) * self.jvp_config[1],
        )
        jvp_args = (
            model_partial,
            (z, t, r),
            grad_params,
        )
        if self.use_autograd:
            return torch.autograd.functional.jvp(*jvp_args, create_graph=True)
        return torch.func.jvp(*jvp_args)


class Normalizer(nn.Module):
    """Base class for normalization that remembers parameters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MinMaxNormalizer(Normalizer):
    """Learns per-channel min/max from x and maps to [-1, 1]."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("data_min", None)
        self.register_buffer("data_max", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2
        reduce_dims = [i for i in range(x.ndim) if i != 1]
        self.data_min = x.amin(dim=reduce_dims).view(-1, 1, 1)
        self.data_max = x.amax(dim=reduce_dims).view(-1, 1, 1)

        scale = (self.data_max - self.data_min).clamp_min(self.eps)
        x = (x - self.data_min) / scale  # [min,max] --> [0,1]
        return x * 2.0 - 1.0  # [0, 1] --> [-1,1]

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1.0, 1.0)
        scale = (self.data_max - self.data_min).clamp_min(self.eps)
        return (x + 1.0) * 0.5 * scale + self.data_min


class MeanStdNormalizer(Normalizer):
    """Learns per-channel mean/std from x and standardizes."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", None)
        self.register_buffer("std", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2
        reduce_dims = [i for i in range(x.ndim) if i != 1]
        self.mean = x.mean(dim=reduce_dims).view(-1, 1, 1)
        self.std = x.std(dim=reduce_dims, unbiased=False).view(-1, 1, 1)
        return (x - self.mean) / self.std.clamp_min(self.eps)

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class CFG:
    def __init__(
        self, cfg_prob: float, cfg_scale: float, use_cond: bool, num_classes: int = None
    ):
        self.cfg_prob = cfg_prob
        self.cfg_scale = cfg_scale
        self.use_cond = use_cond
        self.num_classes = num_classes

    def __call__(
        self,
        v: torch.Tensor,
        label: torch.Tensor,
        model: callable,
        t: float,
        z: torch.tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_cond and self.cfg_prob > 0.0 and self.num_classes is not None:
            uncond = torch.ones_like(label) * self.num_classes

            cfg_mask = torch.rand_like(label.float()) > self.cfg_prob

            masked_labels = torch.where(cfg_mask, label, uncond)

            with torch.no_grad():
                u_t = model(z, t, t, uncond)
                v_hat = self.cfg_scale * v + (1 - self.cfg_scale) * u_t
                v_hat = torch.where(cfg_mask.reshape(-1, 1, 1, 1).bool(), v, v_hat)
        else:
            v_hat = v
            masked_labels = label
        return v_hat, masked_labels


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def cycle(iterable):
    """Infinite dataloader iterator."""
    while True:
        yield from iterable
