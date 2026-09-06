"""VQ-VAE: learn a discrete tokenizer, then learn its token prior.

Stage 1 is the original three-way VQ objective:

    reconstruction + codebook loss + commitment loss

The decoder sees a nearest code in the forward pass, while the straight-
through estimator copies its gradient to the encoder. Stage 2 freezes the
entire tokenizer and trains a class-conditional causal PixelCNN on the 16x16
integer grid. Thus reconstruction and generation remain visibly different
paths. Under the stage-one fixed uniform prior, every position's KL is the
parameter-independent constant ``log(codebook_size)`` and is omitted.

Three stride-2 encoder blocks compress each 128x128 image to 16x16 tokens.
This teaching-scale compromise keeps ancestral PixelCNN sampling at 256
positions without widening the original channel and codebook defaults.

The tokenizer file is a stage handoff, not a training-resume checkpoint. Each
stage saves only final weights, constructor metadata, and the tokenizer identity
needed to prevent pairing a prior with different tokenizer weights.

Data:
    data/celeba (official train and validation splits), prepared by
    tool_scripts/download_dataset.py --dataset celeba.

Outputs:
    output/vae/vq_vae/tokenizer_*.png: reconstruction comparisons
    output/vae/vq_vae/prior_*.png: one PixelCNN sample per Smiling label
    output/vae/vq_vae/tokenizer.pth: final tokenizer checkpoint
    output/vae/vq_vae/pixelcnn_prior.pth: final token-prior checkpoint

Training data -- CelebA-128:
Training images:             162,770
Validation images:            19,867
Batch size:                       64
Samples per epoch:           162,752 (2,543 full batches; drop_last=True)
Tokenizer epochs:                 20
Prior epochs:                     20
Optimizer updates:            50,860 tokenizer / 50,860 prior
The tokenizer remains label-free; the frozen-token PixelCNN uses the binary
Smiling attribute. The final 18 shuffled images are omitted per epoch.
Both splits center-crop aligned faces to 178x178, resize to 128x128, and use
no random horizontal flips.

Default dimensions:
Training input:             128x128 RGB
Generated image:            128x128 RGB
Latent token grid:            16x16 indices (512-entry codebook)

Model size:
VQ-VAE tokenizer:              1.71 M parameters
Conditional PixelCNN prior:    1.84 M parameters (tokenizer frozen)
Stored total:                  3.55 M parameters

Examples:
    python genai/2.0_variational_autoencoder/6.0_vq_vae.py --smoke-test
    python genai/2.0_variational_autoencoder/6.0_vq_vae.py --stage tokenizer
    python genai/2.0_variational_autoencoder/6.0_vq_vae.py --stage prior
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.celeba import (
    CELEBA_SMILING_ATTRIBUTE,
    CELEBA_SMILING_CLASSES,
    make_aligned_celeba_train_validation_loaders,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import (
    model_state_fingerprint,
    save_model_weights,
)
from dl_utils.vae.quantization import VQVAE, TokenUsageAccumulator
from dl_utils.vae.token_prior import (
    PixelCNNPrior,
    evaluate_pixelcnn_prior,
    make_fixed_class_labels,
    sample_pixelcnn_prior_images,
    train_pixelcnn_prior_epoch,
)

PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "celeba"
IMAGE_SIZE = 128
NUM_CLASSES = len(CELEBA_SMILING_CLASSES)
DOWNSAMPLE_STEPS = 3
LATENT_GRID_SIZE = IMAGE_SIZE // (2**DOWNSAMPLE_STEPS)
TOKENS_PER_IMAGE = LATENT_GRID_SIZE**2
SAMPLES_PER_CLASS = 1


def tokenizer_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    quantizer_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distortion = F.mse_loss(reconstruction, x)
    return distortion + quantizer_loss, distortion.detach()


def smoke_test() -> None:
    torch.manual_seed(7)
    tokenizer = VQVAE(
        hidden_channels=32,
        embedding_dim=16,
        codebook_size=32,
        downsample_steps=DOWNSAMPLE_STEPS,
    )
    prior = PixelCNNPrior(32, hidden_channels=24, layers=3, num_classes=NUM_CLASSES)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).clamp(-1, 1)
    labels = torch.tensor([0, NUM_CLASSES - 1])
    reconstruction, indices, quantizer_loss, diagnostics = tokenizer(x)
    loss, _ = tokenizer_loss(reconstruction, x, quantizer_loss)
    loss.backward()
    assert tokenizer.encoder.net[0].weight.grad is not None
    tokenizer.zero_grad(set_to_none=True)
    tokenizer.eval().requires_grad_(False)
    prior_loss = train_pixelcnn_prior_epoch(
        tokenizer,
        prior,
        [(x, labels)],
        torch.optim.Adam(prior.parameters(), lr=2e-4),
        torch.device("cpu"),
    )
    assert all(parameter.grad is None for parameter in tokenizer.parameters())
    assert prior.embedding.weight.grad is not None
    decoded = tokenizer.decode_indices(indices)
    with torch.inference_mode():
        logits = prior(indices, labels=labels)
        sampled = prior.sample(
            1,
            LATENT_GRID_SIZE,
            LATENT_GRID_SIZE,
            device=torch.device("cpu"),
            labels=torch.tensor([0]),
        )
        generated = tokenizer.decode_indices(sampled)
    assert reconstruction.shape == x.shape == decoded.shape
    assert generated.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert indices.shape == (2, LATENT_GRID_SIZE, LATENT_GRID_SIZE)
    assert logits.shape == (2, 32, LATENT_GRID_SIZE, LATENT_GRID_SIZE)
    print(
        f"smoke test passed: tokenizer={loss.item():.3f}, "
        f"prior={prior_loss:.3f}, "
        f"PPL={diagnostics['perplexity'].item():.2f}, "
        f"active={diagnostics['active_codes'].item()}/32"
    )


@torch.inference_mode()
def evaluate_tokenizer(
    model: VQVAE,
    loader: DataLoader,
    *,
    vocabulary_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    distortion = 0.0
    quantization_mse = 0.0
    examples = 0
    usage = TokenUsageAccumulator(vocabulary_size)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        reconstruction, indices, _, diagnostics = model(x)
        distortion += float(F.mse_loss(reconstruction, x)) * x.shape[0]
        quantization_mse += float(diagnostics["quantization_mse"]) * x.shape[0]
        usage.update(indices)
        examples += x.shape[0]
    statistics = usage.statistics()
    entropy_bits = float(statistics["token_entropy_nats"]) / math.log(2)
    return {
        "mse": distortion / examples,
        "quantization_mse": quantization_mse / examples,
        "perplexity": float(statistics["perplexity"]),
        "active_codes": float(statistics["active_codes"]),
        "usage_fraction": float(statistics["usage_fraction"]),
        "marginal_entropy_bits_per_token": entropy_bits,
        "marginal_entropy_bits_per_image": TOKENS_PER_IMAGE * entropy_bits,
        "fixed_length_bits_per_image": (
            TOKENS_PER_IMAGE * math.ceil(math.log2(vocabulary_size))
        ),
    }


def train_tokenizer(
    args: argparse.Namespace,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> VQVAE:
    config = {
        "hidden_channels": args.hidden_channels,
        "embedding_dim": args.embedding_dim,
        "codebook_size": args.codebook_size,
        "commitment": args.commitment,
        "downsample_steps": DOWNSAMPLE_STEPS,
    }
    model = VQVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fixed_bits = TOKENS_PER_IMAGE * math.ceil(math.log2(args.codebook_size))
    print(f"fixed-length upper bound={fixed_bits} bits/image")
    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        sums = torch.zeros(4, device=device)
        examples = 0
        usage = TokenUsageAccumulator(args.codebook_size)
        preview = None
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            reconstruction, indices, quantizer_loss, diagnostics = model(x)
            loss, distortion = tokenizer_loss(reconstruction, x, quantizer_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = [
                loss.detach(),
                distortion,
                diagnostics["codebook_loss"],
                diagnostics["commitment_loss"],
            ]
            sums += torch.stack(values) * x.shape[0]
            usage.update(indices)
            examples += x.shape[0]
            preview = (x[:8].detach(), reconstruction[:8].detach())
        if preview is None:
            raise ValueError("training loader produced no batches; reduce batch size")
        means = (sums / examples).tolist()
        epoch_usage = usage.statistics()
        print(
            f"tokenizer {epoch:03d}: loss={means[0]:.4f}, D={means[1]:.4f}, "
            f"code={means[2]:.4f}, commit={means[3]:.4f}, "
            f"PPL={epoch_usage['perplexity'].item():.1f}, "
            f"active={epoch_usage['active_codes'].item()}/{args.codebook_size}"
        )
        if epoch == 1 or epoch % args.sample_every == 0:
            save_image(
                torch.cat(preview).mul(0.5).add(0.5),
                out_dir / f"tokenizer_{epoch:03d}.png",
                nrow=8,
            )
    validation = evaluate_tokenizer(
        model,
        validation_loader,
        vocabulary_size=args.codebook_size,
        device=device,
    )
    interface_id = model_state_fingerprint(model)
    save_model_weights(
        model,
        out_dir / "tokenizer.pth",
        metadata={
            "model_name": "vq_vae_tokenizer",
            "interface_id": interface_id,
            "model_config": config,
            "dataset": "celeba",
            "image_size": IMAGE_SIZE,
        },
    )
    print(
        f"validation tokenizer: MSE={validation['mse']:.4f}, "
        f"entropy={validation['marginal_entropy_bits_per_token']:.3f} "
        f"bits/token, active={validation['active_codes']:.0f}/"
        f"{args.codebook_size}"
    )
    return model.eval().requires_grad_(False)


def load_tokenizer(checkpoint_path: Path, device: torch.device) -> VQVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "vq_vae_tokenizer":
        raise ValueError("checkpoint is not a VQ-VAE tokenizer")
    if (
        checkpoint.get("dataset") != "celeba"
        or checkpoint.get("image_size") != IMAGE_SIZE
    ):
        raise ValueError("tokenizer is not the CelebA-128 configuration")
    model = VQVAE(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("interface_id") != model_state_fingerprint(model):
        raise ValueError(
            "tokenizer checkpoint has no valid interface identity; retrain stage 1"
        )
    return model.eval().requires_grad_(False)


def train_prior(
    args: argparse.Namespace,
    tokenizer: VQVAE,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    tokenizer.eval().requires_grad_(False)
    vocabulary_size = tokenizer.quantizer.codebook_size
    prior = PixelCNNPrior(
        vocabulary_size,
        hidden_channels=args.prior_hidden_channels,
        layers=args.prior_layers,
        num_classes=NUM_CLASSES,
    ).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.prior_lr)
    for epoch in range(1, args.prior_epochs + 1):
        nll = train_pixelcnn_prior_epoch(
            tokenizer,
            prior,
            train_loader,
            optimizer,
            device,
        )
        print(
            f"prior {epoch:03d}: train nll={nll:.4f}, "
            f"train bits/token={nll / math.log(2):.3f}"
        )
        if epoch == 1 or epoch % args.sample_every == 0:
            prior.eval()
            labels = make_fixed_class_labels(NUM_CLASSES, SAMPLES_PER_CLASS, device)
            samples = sample_pixelcnn_prior_images(
                tokenizer,
                prior,
                labels,
                grid_size=LATENT_GRID_SIZE,
                device=device,
                temperature=args.temperature,
            )
            save_image(
                samples.mul(0.5).add(0.5),
                out_dir / f"prior_{epoch:03d}.png",
                nrow=NUM_CLASSES,
            )
    validation = evaluate_pixelcnn_prior(
        tokenizer,
        prior,
        validation_loader,
        tokens_per_image=TOKENS_PER_IMAGE,
        device=device,
    )
    tokenizer_interface_id = model_state_fingerprint(tokenizer)
    prior_config = {
        "vocabulary_size": vocabulary_size,
        "hidden_channels": args.prior_hidden_channels,
        "layers": args.prior_layers,
        "num_classes": NUM_CLASSES,
    }
    save_model_weights(
        prior,
        out_dir / "pixelcnn_prior.pth",
        metadata={
            "model_name": "vq_vae_pixelcnn_prior",
            "model_config": prior_config,
            "tokenizer_interface_id": tokenizer_interface_id,
            "dataset": "celeba",
            "image_size": IMAGE_SIZE,
            "conditioning": "class_conditional",
            "attribute": CELEBA_SMILING_ATTRIBUTE,
            "class_names": list(CELEBA_SMILING_CLASSES),
        },
    )
    print(
        f"validation prior: {validation['bits_per_token']:.3f} "
        f"bits/token, {validation['bits_per_image']:.1f} bits/image"
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vae" / "vq_vae"
    train_loader, validation_loader = make_aligned_celeba_train_validation_loaders(
        args.data_dir,
        IMAGE_SIZE,
        args.batch_size,
        device,
        num_workers=args.workers,
        attribute=CELEBA_SMILING_ATTRIBUTE,
    )
    checkpoint_path = out_dir / "tokenizer.pth"
    if args.stage in {"tokenizer", "all"}:
        reset_dir(str(out_dir))
        tokenizer = train_tokenizer(
            args, train_loader, validation_loader, device, out_dir
        )
    else:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("train --stage tokenizer before the prior")
        tokenizer = load_tokenizer(checkpoint_path, device)
        # The tokenizer is an input to this stage; keep its handoff checkpoint.
        tokenizer_checkpoint = checkpoint_path.read_bytes()
        reset_dir(str(out_dir))
        checkpoint_path.write_bytes(tokenizer_checkpoint)
        del tokenizer_checkpoint
    if args.stage in {"prior", "all"}:
        train_prior(
            args,
            tokenizer,
            train_loader,
            validation_loader,
            device,
            out_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stage", choices=("tokenizer", "prior", "all"), default="all")
    parser.add_argument("--tokenizer-epochs", type=int, default=20)
    parser.add_argument("--prior-epochs", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--prior-hidden-channels", type=int, default=128)
    parser.add_argument("--prior-layers", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
