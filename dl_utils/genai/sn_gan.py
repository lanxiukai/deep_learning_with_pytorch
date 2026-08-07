"""Conditional SN-GAN blocks, models, losses, and image helpers.

This compact baseline keeps class conditioning fixed across the SN-GAN,
SAGAN, and BigGAN lessons. The generator concatenates one class embedding to
the latent vector, while the discriminator uses projection conditioning.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class SNGeneratorResidualBlock(nn.Module):
    """Spectral-normalized residual block that upsamples by a factor of two."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs):
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        hidden = F.relu(self.norm1(inputs), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(self.norm2(hidden), inplace=True))
        return hidden + residual


class SNDiscriminatorResidualBlock(nn.Module):
    """Spectral-normalized residual block that downsamples by a factor of two."""

    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        self.first = first
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, inputs):
        hidden = inputs if self.first else F.relu(inputs, inplace=False)
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(hidden, inplace=True))
        hidden = F.avg_pool2d(hidden, 2)

        residual = F.avg_pool2d(self.skip(inputs), 2)
        return hidden + residual


class SNGenerator(nn.Module):
    """Compact 64x64 conditional SN-GAN generator.

    Class information enters once at the generator input. The ``attention``
    attribute is an identity layer so the SAGAN lesson can replace exactly one
    component without changing this forward method.
    """

    def __init__(
        self,
        z_dim=120,
        num_classes=10,
        base_channels=32,
        condition_dim=128,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.class_embedding = nn.Embedding(num_classes, condition_dim)
        self.input = spectral_norm(
            nn.Linear(z_dim + condition_dim, base_channels * 8 * 4 * 4)
        )
        self.block1 = SNGeneratorResidualBlock(
            base_channels * 8, base_channels * 4
        )
        self.block2 = SNGeneratorResidualBlock(
            base_channels * 4, base_channels * 2
        )
        self.attention = nn.Identity()
        self.block3 = SNGeneratorResidualBlock(
            base_channels * 2, base_channels
        )
        self.block4 = SNGeneratorResidualBlock(base_channels, base_channels)
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, noise, labels):
        # noise shape: (B, 120), labels shape: (B,)
        condition = self.class_embedding(labels)  # (B, 128)
        hidden = self.input(torch.cat([noise, condition], dim=1))  # (B, 248) -> (B, 4096)
        hidden = hidden.view(noise.shape[0], -1, 4, 4)  # (B, 256, 4, 4)
        hidden = self.block1(hidden)     # (B, 128, 8, 8)
        hidden = self.block2(hidden)     # (B, 64, 16, 16)
        hidden = self.attention(hidden)  # (B, 64, 16, 16)
        hidden = self.block3(hidden)     # (B, 32, 32, 32)
        hidden = self.block4(hidden)     # (B, 32, 64, 64)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))  # (B, 3, 64, 64)


class ProjectionSNDiscriminator(nn.Module):
    """Compact 64x64 SN-GAN discriminator with projection conditioning."""

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        channels = base_channels
        self.block1 = SNDiscriminatorResidualBlock(
            3, channels, first=True
        )
        self.block2 = SNDiscriminatorResidualBlock(channels, channels * 2)
        self.attention = nn.Identity()
        self.block3 = SNDiscriminatorResidualBlock(
            channels * 2, channels * 4
        )
        self.block4 = SNDiscriminatorResidualBlock(
            channels * 4, channels * 8
        )
        self.output = spectral_norm(nn.Linear(channels * 8, 1))
        self.class_embedding = spectral_norm(
            nn.Embedding(num_classes, channels * 8)
        )

    def forward(self, images, labels):
        # images shape: (B, 3, 64, 64)，labels shape: (B,)
        hidden = self.block1(images)     # (B, 32, 32, 32)
        hidden = self.block2(hidden)     # (B, 64, 16, 16)
        hidden = self.attention(hidden)  # (B, 64, 16, 16)
        hidden = self.block3(hidden)     # (B, 128, 8, 8)
        hidden = self.block4(hidden)     # (B, 256, 4, 4)
        hidden = F.relu(hidden, inplace=True).sum(dim=(2, 3))  # (B, 256)

        unconditional = self.output(hidden).squeeze(1)
        # Projection discriminator: D(x, y) = f(h(x)) + <e(y), h(x)>.
        projection = (self.class_embedding(labels) * hidden).sum(dim=1)
        return unconditional + projection  # (B,)


def discriminator_hinge_loss(real_scores, fake_scores):
    """Return the discriminator hinge loss."""
    return F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()


def generator_hinge_loss(fake_scores):
    """Return the non-saturating hinge objective used by the generator."""
    return -fake_scores.mean()


def denormalize(images):
    """Map images from [-1, 1] to [0, 1]."""
    return images.mul(0.5).add(0.5).clamp(0, 1)


class _SpectralNormScratch:
    """Forward pre-hook that computes a spectrally normalized weight.

    The original parameter is kept as ``<name>_orig``. Two one-dimensional
    buffers, ``<name>_u`` and ``<name>_v``, approximate the leading left and
    right singular vectors of the flattened weight matrix.
    """

    def __init__(self, name, n_power_iterations, dim, eps):
        if n_power_iterations <= 0:
            raise ValueError(
                "n_power_iterations must be positive, "
                f"but got {n_power_iterations}"
            )
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps

    def _reshape_weight_to_matrix(self, weight):
        """Move the output dimension first, then flatten all input axes."""
        if self.dim != 0:
            axes = [self.dim]
            axes.extend(axis for axis in range(weight.ndim) if axis != self.dim)
            weight = weight.permute(*axes)
        return weight.reshape(weight.shape[0], -1)

    def _normalize(self, vector):
        """Return a unit vector while avoiding division by a tiny norm."""
        norm = torch.linalg.vector_norm(vector).clamp_min(self.eps)
        return vector / norm

    def _compute_weight(self, module):
        weight_orig = getattr(module, f"{self.name}_orig")
        u = getattr(module, f"{self.name}_u")
        v = getattr(module, f"{self.name}_v")
        weight_matrix = self._reshape_weight_to_matrix(weight_orig)

        if module.training:
            # Power iteration alternates between the two singular-vector
            # equations: v <- normalize(W^T u), u <- normalize(W v).
            # These estimates are buffers, not trainable parameters.
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v.copy_(self._normalize(torch.mv(weight_matrix.t(), u)))
                    u.copy_(self._normalize(torch.mv(weight_matrix, v)))

            # A discriminator is often called for both real and fake samples
            # before one backward pass. Clones prevent the second call's
            # in-place power iteration from invalidating the first graph.
            u_for_sigma = u.clone()
            v_for_sigma = v.clone()
        else:
            u_for_sigma = u
            v_for_sigma = v

        # For the leading singular vectors, sigma = u^T W v approximates the
        # largest singular value. Division makes the weight's spectral norm
        # approximately one while gradients still flow to weight_orig.
        sigma = torch.dot(
            u_for_sigma,
            torch.mv(weight_matrix, v_for_sigma),
        )
        return weight_orig / sigma

    def __call__(self, module, inputs):
        """Replace the public weight immediately before every forward call."""
        setattr(module, self.name, self._compute_weight(module))

    def apply(self, module):
        """Move the parameter, initialize power iteration, and install hook."""
        for hook in module._forward_pre_hooks.values():
            if (
                isinstance(hook, _SpectralNormScratch)
                and hook.name == self.name
            ):
                raise RuntimeError(
                    "Cannot register spectral_norm_scratch twice for "
                    f"parameter {self.name!r}"
                )

        weight = module._parameters.get(self.name)
        if weight is None:
            raise ValueError(
                "spectral_norm_scratch requires an initialized parameter "
                f"named {self.name!r}"
            )
        if isinstance(weight, nn.parameter.UninitializedParameter):
            raise ValueError(
                "spectral_norm_scratch does not support uninitialized "
                "parameters; run a dummy forward first"
            )

        with torch.no_grad():
            weight_matrix = self._reshape_weight_to_matrix(weight)
            height, width = weight_matrix.shape
            u = self._normalize(weight.new_empty(height).normal_())
            v = self._normalize(weight.new_empty(width).normal_())

        # Optimizers should update the unconstrained weight_orig. The public
        # weight becomes an ordinary Tensor recomputed by the pre-hook.
        delattr(module, self.name)
        module.register_parameter(f"{self.name}_orig", weight)
        setattr(module, self.name, weight.detach())
        module.register_buffer(f"{self.name}_u", u)
        module.register_buffer(f"{self.name}_v", v)
        module.register_forward_pre_hook(self)


def spectral_norm_scratch(
    module,
    name="weight",
    n_power_iterations=1,
    eps=1e-12,
    dim=None,
):
    """Apply spectral normalization using an explicit power-iteration hook.

    This teaching implementation mirrors the commonly used
    :func:`torch.nn.utils.spectral_norm` interface and can replace it for the
    ``Linear``, ``Conv2d``, and ``Embedding`` layers in this module. It skips
    legacy state-dict migration and hook-removal utilities.

    Args:
        module: Module that owns the parameter to normalize.
        name: Name of the weight parameter.
        n_power_iterations: Power-iteration steps performed per training call.
        eps: Lower bound used when normalizing the singular-vector estimates.
        dim: Weight axis representing outputs. Defaults to 1 for transposed
            convolutions and 0 for other modules.

    Returns:
        The same module, modified in place with a forward pre-hook.
    """
    if dim is None:
        transposed_convolutions = (
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )
        dim = 1 if isinstance(module, transposed_convolutions) else 0

    hook = _SpectralNormScratch(name, n_power_iterations, dim, eps)
    hook.apply(module)
    return module
