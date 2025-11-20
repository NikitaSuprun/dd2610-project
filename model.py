from einops import rearrange
import torch
import torch.nn as nn

from utils import JVP, Loss, Normalizer, TRSampler, stopgrad


class MeanFlow:
    def __init__(
        self,
        model: nn.Module,
        normalizer: Normalizer,
        loss_fn: Loss,
        t_r_sampler: TRSampler,
        jvp_fn: JVP,
        channels=1,
        image_size=32,
        num_classes=10,
        cfg_ratio=0.0,
        cfg_scale=0.0,
    ):
        self.model = model
        self.normalizer = normalizer

        self.loss_fn = loss_fn
        self.t_r_sampler = t_r_sampler
        self.jvp_fn = jvp_fn

        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.cfg_ratio = cfg_ratio
        self.cfg_scale = cfg_scale

    def loss(self, x: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        t, r = self.t_r_sampler.sample(batch_size, device)

        x = self.normalizer.forward(x)
        e = torch.randn_like(x)

        # Reshape t and r for broadcasting with image tensors (B, C, H, W)
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        z = (1 - t_) * x + t_ * e
        v = e - x

        if labels is not None:
            v, labels = self._cfg(v, labels, self.model, t, z)
        u, dudt = self.jvp_fn(self.model, z, t, r, labels, v)
        u_tgt = v - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        return self.loss_fn(error)

    def _cfg(self, v, labels, model, t, z):
        if self.cfg_ratio <= 0.0:
            return v, labels

        uncond = torch.ones_like(labels) * self.num_classes
        cfg_mask = torch.rand_like(labels.float()) < self.cfg_ratio
        masked_labels = torch.where(cfg_mask, uncond, labels)

        if self.cfg_scale <= 0.0:
            return v, masked_labels

        with torch.no_grad():
            u_t = model(z, t, t, uncond)
            v_hat = self.cfg_scale * v + (1 - self.cfg_scale) * u_t

            if self.use_cond:
                cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                v_hat = torch.where(cfg_mask, v, v_hat)
        return v_hat, masked_labels

    @torch.no_grad()
    def sample_each_class(self, n_per_class, classes=None, sample_steps=5):
        """
        Generate samples for each class.

        Args:
            n_per_class: Number of samples per class
            classes: List of class indices to generate (default: all classes)
            sample_steps: Number of sampling steps

        Returns:
            Generated images (N, C, H, W) in [0, 1] range
        """
        self.model.eval()
        device = torch.device("cuda")

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        z = torch.randn(
            c.shape[0], self.channels, self.image_size, self.image_size, device=device
        )

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            t_ = rearrange(t, "b -> b 1 1 1")
            r_ = rearrange(r, "b -> b 1 1 1")

            v = self.model(z, t, r, c)
            z = z - (t_ - r_) * v

        z = self.normalizer.unnorm(z)
        return z
