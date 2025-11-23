from dataclasses import dataclass
from enum import Enum
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 backport

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
    n_epochs: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    gamma: float
    c: float
    sample_ratio: float
    sampler_type: str
    sample_params: dict
    log_step: int
    sample_every_n_epochs: int
    checkpoint_every_n_epochs: int
    histogram_step: int
    mixed_precision: str
    jvp_use_autograd: bool
    n_samples_per_class: int
    sampling_steps: int
    sample_grid_nrow: int
    cfg_ratio: float
    cfg_scale: float
    use_gradient_checkpointing: bool


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
            n_epochs=training_data.get("n_epochs", 100),
            num_workers=training_data.get("num_workers", 8),
            learning_rate=training_data.get("learning_rate", 0.001),
            weight_decay=training_data.get("weight_decay", 0.0001),
            gamma=training_data.get("gamma", 0.5),
            c=training_data.get("c", 1.0),
            sample_ratio=training_data.get("sample_ratio", 0.5),
            sampler_type=training_data.get("sampler_type", "uniform"),
            sample_params=training_data.get("sample_params", {}),
            log_step=training_data.get("log_step", 500),
            sample_every_n_epochs=training_data.get("sample_every_n_epochs", 5),
            checkpoint_every_n_epochs=training_data.get("checkpoint_every_n_epochs", 5),
            histogram_step=training_data.get("histogram_step", 2000),
            mixed_precision=training_data.get("mixed_precision", "fp16"),
            jvp_use_autograd=training_data.get("jvp_use_autograd", False),
            n_samples_per_class=training_data.get("n_samples_per_class", 1),
            sampling_steps=training_data.get("sampling_steps", 5),
            sample_grid_nrow=training_data.get("sample_grid_nrow", 10),
            cfg_ratio=training_data.get("cfg_ratio", 0.0),
            cfg_scale=training_data.get("cfg_scale", 0.0),
            use_gradient_checkpointing=training_data.get("use_gradient_checkpointing", False),
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
    """Maps x from [0, 1] to [-1, 1]."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assumes x is in [0, 1] range (e.g., from ToTensor())."""
        return x * 2.0 - 1.0  # [0, 1] --> [-1, 1]

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        """Maps from [-1, 1] back to [0, 1]."""
        x = x.clamp(-1.0, 1.0)
        return (x + 1.0) * 0.5  # [-1, 1] --> [0, 1]


class MeanStdNormalizer(Normalizer):
    """Standardizes using fixed mean and std values."""

    def __init__(self, mean=(0.5,), std=(0.5,), eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Register as buffers so they move with the model to the correct device
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize using fixed mean and std."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device).clamp_min(self.eps)

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using fixed mean and std."""
        return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def cycle(iterable):
    """Infinite dataloader iterator."""
    while True:
        yield from iterable
