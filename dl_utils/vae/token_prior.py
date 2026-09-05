"""Compact causal token priors shared by VQ-VAE, FSQ, and VQGAN."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _validate_class_labels(
    labels: Tensor | None,
    *,
    batch_size: int,
    num_classes: int,
) -> Tensor | None:
    if num_classes == 0:
        if labels is not None:
            raise ValueError("this prior was built without class conditioning")
        return None
    if labels is None:
        raise ValueError("class-conditional prior requires labels")
    if labels.dtype != torch.long or labels.shape != (batch_size,):
        raise ValueError("labels must be a torch.long tensor with shape [B]")
    return labels


def make_fixed_class_labels(
    num_classes: int,
    samples_per_class: int,
    device: torch.device,
) -> Tensor:
    """Return adjacent, balanced labels for readable generated-image grids."""
    if min(num_classes, samples_per_class) < 1:
        raise ValueError("num_classes and samples_per_class must be positive")
    return torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)


class MaskedConv2d(nn.Conv2d):
    """PixelCNN mask: type A excludes the current token; B includes it."""

    def __init__(self, mask_type: str, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        mask = torch.ones_like(self.weight)
        center_h = self.kernel_size[0] // 2
        center_w = self.kernel_size[1] // 2
        mask[:, :, center_h + 1 :, :] = 0
        first_blocked = center_w if mask_type == "A" else center_w + 1
        mask[:, :, center_h, first_blocked:] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNNPrior(nn.Module):
    """Small unconditional or spatially conditioned causal image-token prior."""

    def __init__(
        self,
        vocabulary_size: int,
        *,
        hidden_channels: int = 128,
        layers: int = 7,
        condition_channels: int = 0,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        if vocabulary_size < 2 or layers < 1 or num_classes < 0:
            raise ValueError("invalid PixelCNN configuration")
        self.vocabulary_size = vocabulary_size
        self.condition_channels = condition_channels
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocabulary_size, hidden_channels)
        self.class_embedding = (
            nn.Embedding(num_classes, hidden_channels) if num_classes > 0 else None
        )
        self.condition_projection = (
            nn.Conv2d(condition_channels, hidden_channels, 1)
            if condition_channels > 0
            else None
        )
        blocks: list[nn.Module] = [
            MaskedConv2d("A", hidden_channels, hidden_channels, 7, padding=3),
            nn.ReLU(inplace=True),
        ]
        for _ in range(layers - 1):
            blocks.extend(
                [
                    MaskedConv2d("B", hidden_channels, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        self.causal = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, vocabulary_size, 1),
        )

    def forward(
        self,
        indices: Tensor,
        condition: Tensor | None = None,
        *,
        labels: Tensor | None = None,
    ) -> Tensor:
        hidden = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        hidden = self.causal(hidden)
        labels = _validate_class_labels(
            labels,
            batch_size=indices.shape[0],
            num_classes=self.num_classes,
        )
        if self.class_embedding is not None:
            hidden = hidden + self.class_embedding(labels)[:, :, None, None]
        if self.condition_projection is None:
            if condition is not None:
                raise ValueError("this prior was built without conditioning")
        else:
            if condition is None:
                raise ValueError("conditional prior requires a spatial condition")
            if (
                condition.shape[0] != indices.shape[0]
                or condition.shape[2:] != indices.shape[1:]
            ):
                raise ValueError("condition must align with the index grid")
            hidden = hidden + self.condition_projection(condition)
        return self.head(hidden)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        condition: Tensor | None = None,
        labels: Tensor | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        indices = torch.zeros(count, height, width, dtype=torch.long, device=device)
        for row in range(height):
            for column in range(width):
                logits = (
                    self(indices, condition, labels=labels)[:, :, row, column]
                    / temperature
                )
                indices[:, row, column] = torch.multinomial(
                    logits.softmax(dim=1), 1
                ).squeeze(1)
        return indices


class CausalTransformerPrior(nn.Module):
    """Teacher-forced causal Transformer over a fixed-length token sequence."""

    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        *,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.0,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        if vocabulary_size < 2 or sequence_length < 1 or num_classes < 0:
            raise ValueError("invalid vocabulary or sequence length")
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.bos_token = vocabulary_size
        self.token_embedding = nn.Embedding(vocabulary_size + 1, model_dim)
        self.class_embedding = (
            nn.Embedding(num_classes, model_dim) if num_classes > 0 else None
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, sequence_length, model_dim) / math.sqrt(model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocabulary_size, bias=False)

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, input_tokens: Tensor, labels: Tensor | None = None) -> Tensor:
        if input_tokens.ndim != 2:
            raise ValueError("input_tokens must have shape [B, T]")
        length = input_tokens.shape[1]
        if length < 1 or length > self.sequence_length:
            raise ValueError("input sequence length is outside the configured range")
        hidden = self.token_embedding(input_tokens)
        hidden = hidden + self.position_embedding[:, :length]
        labels = _validate_class_labels(
            labels,
            batch_size=input_tokens.shape[0],
            num_classes=self.num_classes,
        )
        if self.class_embedding is not None:
            hidden = hidden + self.class_embedding(labels)[:, None, :]
        hidden = self.transformer(
            hidden, mask=self._causal_mask(length, input_tokens.device)
        )
        return self.head(self.normalization(hidden))

    def teacher_forcing(
        self,
        indices: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return logits and targets for ``[B,H,W]`` or ``[B,T]`` indices."""
        targets = indices.flatten(1)
        if targets.shape[1] != self.sequence_length:
            raise ValueError("token grid does not match sequence_length")
        bos = targets.new_full((targets.shape[0], 1), self.bos_token)
        inputs = torch.cat((bos, targets[:, :-1]), dim=1)
        return self(inputs, labels), targets

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        labels: Tensor | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        sequence = torch.full(
            (count, 1), self.bos_token, dtype=torch.long, device=device
        )
        generated: list[Tensor] = []
        for _ in range(self.sequence_length):
            logits = self(sequence, labels)[:, -1] / temperature
            token = torch.multinomial(logits.softmax(dim=1), 1)
            generated.append(token)
            if len(generated) < self.sequence_length:
                sequence = torch.cat((sequence, token), dim=1)
        return torch.cat(generated, dim=1)


__all__ = [
    "CausalTransformerPrior",
    "MaskedConv2d",
    "PixelCNNPrior",
    "make_fixed_class_labels",
]
