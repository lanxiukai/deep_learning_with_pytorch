"""Compact conditional latent-variable models shared by the cVAE lessons."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl_utils.vae.vae_common import (
    reparameterize_logvar,
    split_gaussian_parameters,
)


def _validate_labels(labels: Tensor, batch_size: int, num_classes: int) -> None:
    if labels.shape != (batch_size,):
        raise ValueError("labels must have shape [batch]")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long class indices")
    if labels.numel() and (
        int(labels.min()) < 0 or int(labels.max()) >= num_classes
    ):
        raise ValueError("labels contain an out-of-range class index")


class ConditionalDecoder32(nn.Module):
    """Decode a latent and class representation into a Bernoulli mean."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_channels * 4 * 4),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Unflatten(1, (hidden_channels, 4, 4)),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels // 2, 4, 2, 1
            ),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(
                hidden_channels // 2, hidden_channels // 4, 4, 2, 1
            ),
            nn.GroupNorm(4, hidden_channels // 4),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 4, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor, condition: Tensor) -> Tensor:
        if z.shape[:-1] != condition.shape[:-1]:
            raise ValueError("z and condition must share all leading dimensions")
        leading_shape = z.shape[:-1]
        features = torch.cat((z, condition), dim=-1).reshape(
            -1, z.shape[-1] + condition.shape[-1]
        )
        images = self.net(self.input(features))
        return images.reshape(*leading_shape, 1, 32, 32)


class ConditionalVAE32(nn.Module):
    """Class-conditional VAE with explicit prior, posterior, and decoder APIs."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        latent_dim: int = 16,
        condition_dim: int = 32,
        hidden_channels: int = 128,
        learned_prior: bool = True,
    ) -> None:
        super().__init__()
        if hidden_channels % 8:
            raise ValueError("hidden_channels must be divisible by 8")
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_channels = hidden_channels
        self.learned_prior = learned_prior

        # This shared embedding receives gradients from every probability
        # interface.  Generation still supplies only the class index.
        self.condition_embedding = nn.Embedding(num_classes, condition_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels // 4, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 4, hidden_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 4 + condition_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * latent_dim),
        )
        self.prior_network = (
            nn.Sequential(
                nn.Linear(condition_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 2 * latent_dim),
            )
            if learned_prior
            else None
        )
        self.decoder = ConditionalDecoder32(
            latent_dim, condition_dim, hidden_channels
        )

    def condition(self, labels: Tensor) -> Tensor:
        _validate_labels(labels, labels.shape[0], self.num_classes)
        return self.condition_embedding(labels)

    def prior(self, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return p(z | c) parameters without observing a target image."""
        condition = self.condition(labels)
        if self.prior_network is None:
            shape = (*labels.shape, self.latent_dim)
            return (
                condition.new_zeros(shape),
                condition.new_zeros(shape),
            )
        return split_gaussian_parameters(self.prior_network(condition))

    def encode(self, x: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return q(z | x, c) parameters; this is the only target-aware API."""
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError("x must have shape [batch, 1, 32, 32]")
        _validate_labels(labels, x.shape[0], self.num_classes)
        condition = self.condition_embedding(labels)
        features = self.image_encoder(x)
        return split_gaussian_parameters(
            self.posterior(torch.cat((features, condition), dim=1))
        )

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        """Return the Bernoulli mean p(x | z, c)."""
        if z.shape[-1] != self.latent_dim:
            raise ValueError("the final z dimension must equal latent_dim")
        if labels.shape != z.shape[:-1]:
            raise ValueError("labels must match the leading z dimensions")
        flat_labels = labels.reshape(-1)
        _validate_labels(flat_labels, flat_labels.shape[0], self.num_classes)
        condition = self.condition_embedding(flat_labels).reshape(
            *labels.shape, self.condition_dim
        )
        return self.decoder(z, condition)

    def reconstruct(
        self, x: Tensor, labels: Tensor, *, sample: bool = False
    ) -> Tensor:
        q_mu, q_logvar = self.encode(x, labels)
        z = reparameterize_logvar(q_mu, q_logvar) if sample else q_mu
        return self.decode(z, labels)

    def generate(
        self,
        labels: Tensor,
        *,
        sample: bool = True,
        epsilon: Tensor | None = None,
    ) -> Tensor:
        """Generate using p(z | c); no target image enters this path."""
        p_mu, p_logvar = self.prior(labels)
        if epsilon is not None:
            if epsilon.shape != p_mu.shape:
                raise ValueError("epsilon must match the conditional prior shape")
            z = p_mu + torch.exp(0.5 * p_logvar) * epsilon
        elif sample:
            z = reparameterize_logvar(p_mu, p_logvar)
        else:
            z = p_mu
        return self.decode(z, labels)

    def forward(
        self, x: Tensor, labels: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        q_mu, q_logvar = self.encode(x, labels)
        p_mu, p_logvar = self.prior(labels)
        z = reparameterize_logvar(q_mu, q_logvar)
        reconstruction = self.decode(z, labels)
        return reconstruction, {
            "z": z,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
        }


class ConditionalMeanDecoder32(nn.Module):
    """Deterministic p(x | c) baseline with no target-aware latent path."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        condition_dim: int = 32,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.hidden_channels = hidden_channels
        self.condition_embedding = nn.Embedding(num_classes, condition_dim)
        self.decoder = ConditionalDecoder32(
            latent_dim=1,
            condition_dim=condition_dim,
            hidden_channels=hidden_channels,
        )

    def generate(self, labels: Tensor) -> Tensor:
        _validate_labels(labels, labels.shape[0], self.num_classes)
        condition = self.condition_embedding(labels)
        zero = condition.new_zeros(labels.shape[0], 1)
        return self.decoder(zero, condition)

    def forward(self, labels: Tensor) -> Tensor:
        return self.generate(labels)


__all__ = [
    "ConditionalMeanDecoder32",
    "ConditionalVAE32",
]
