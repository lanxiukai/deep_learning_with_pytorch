"""FSQ: replace learned vector lookup with fixed scalar quantization.

This is a matched tokenizer branch of VQ-VAE. Relative to
``6.0_vq_vae.py``, the encoder, decoder, straight-through gradient, frozen
tokenizer boundary, and class-conditional second-stage PixelCNN remain. The
quantizer changes:

* each low-dimensional channel is bounded and rounded to fixed levels;
* mixed-radix packing maps the scalar tuple to one integer token;
* no learned codebook, codebook loss, or commitment loss exists.

Removing dead *entries* does not guarantee that the encoder visits every
scalar combination, so usage entropy and reconstruction are still logged.
The 128x128 image path and 16x16 token grid exactly match ``6.0_vq_vae.py`` so
the quantizer remains the controlled algorithmic difference.

The tokenizer file is a stage handoff, not a training-resume checkpoint. Each
stage saves only final weights, constructor metadata, and the tokenizer identity
needed to prevent pairing a prior with different tokenizer weights.

Data:
    data/imagenette-128/{train,val}/<WNID>/*.JPEG, prepared by
    tool_scripts/download_dataset.py --dataset imagenette.

Outputs:
    output/vae/fsq/tokenizer_*.png: reconstruction comparisons
    output/vae/fsq/prior_*.png: one PixelCNN sample per Imagenette class
    output/vae/fsq/tokenizer.pth: final FSQ tokenizer checkpoint
    output/vae/fsq/pixelcnn_prior.pth: final token-prior checkpoint

Training data -- Imagenette-128:
Training images:               9,469
Validation images:             3,925
Batch size:                       64
Samples per epoch:             9,408 (147 full batches; drop_last=True)
Tokenizer epochs:                 20
Prior epochs:                     20
Optimizer updates:             2,940 tokenizer / 2,940 prior
Note: the tokenizer remains label-free; class labels condition only the frozen-
token PixelCNN prior. The final 61 shuffled images are omitted per epoch.

Default dimensions:
Training input:             128x128 RGB
Generated image:            128x128 RGB
Latent token grid:            16x16 indices (1,600 fixed combinations)

Model size:
FSQ tokenizer:                 1.60 M parameters
Conditional PixelCNN prior:    2.12 M parameters (tokenizer frozen)
Stored total:                  3.72 M parameters
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.imagenette import (
    IMAGENETTE_IMAGE_SIZE,
    IMAGENETTE_NUM_CLASSES,
    make_imagenette_train_validation_loaders,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import (
    model_state_fingerprint,
    save_model_weights,
)
from dl_utils.vae.quantization import FSQAutoencoder, TokenUsageAccumulator
from dl_utils.vae.token_prior import (
    PixelCNNPrior,
    evaluate_pixelcnn_prior,
    make_fixed_class_labels,
    sample_pixelcnn_prior_images,
    train_pixelcnn_prior_epoch,
)

PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "imagenette-128"
IMAGE_SIZE = IMAGENETTE_IMAGE_SIZE
NUM_CLASSES = IMAGENETTE_NUM_CLASSES
DOWNSAMPLE_STEPS = 3
LATENT_GRID_SIZE = IMAGE_SIZE // (2**DOWNSAMPLE_STEPS)
TOKENS_PER_IMAGE = LATENT_GRID_SIZE**2
SAMPLES_PER_CLASS = 1


def parse_levels(text: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "levels must be comma-separated integers"
        ) from error
    if not levels or any(level < 2 for level in levels):
        raise argparse.ArgumentTypeError("every level count must be >= 2")
    return levels


def smoke_test() -> None:
    torch.manual_seed(7)
    model = FSQAutoencoder(
        levels=(4, 4, 4),
        hidden_channels=32,
        downsample_steps=DOWNSAMPLE_STEPS,
    )
    prior = PixelCNNPrior(64, hidden_channels=24, layers=3, num_classes=NUM_CLASSES)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).clamp(-1, 1)
    labels = torch.tensor([0, NUM_CLASSES - 1])
    reconstruction, indices, diagnostics = model(x)
    loss = F.mse_loss(reconstruction, x)
    loss.backward()
    prior_loss = F.cross_entropy(
        prior(indices.detach(), labels=labels), indices.detach()
    )
    prior_loss.backward()
    with torch.inference_mode():
        sampled = prior.sample(
            1,
            LATENT_GRID_SIZE,
            LATENT_GRID_SIZE,
            device=torch.device("cpu"),
            labels=torch.tensor([0]),
        )
        generated = model.decode_indices(sampled)

    all_indices = torch.arange(model.quantizer.codebook_size)
    restored = model.quantizer.pack(model.quantizer.unpack(all_indices))
    assert torch.equal(all_indices, restored), "mixed-radix mapping is not reversible"
    assert reconstruction.shape == x.shape
    assert generated.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert indices.shape == (2, LATENT_GRID_SIZE, LATENT_GRID_SIZE)
    assert model.encoder.net[0].weight.grad is not None
    print(
        f"smoke test passed: D={loss.item():.3f}, prior={prior_loss.item():.3f}, "
        f"vocabulary={model.quantizer.codebook_size}, "
        f"PPL={diagnostics['perplexity'].item():.2f}, "
        f"active={diagnostics['active_codes'].item()}"
    )


@torch.inference_mode()
def evaluate_tokenizer(
    model: FSQAutoencoder,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    vocabulary_size = model.quantizer.codebook_size
    distortion = 0.0
    quantization_mse = 0.0
    examples = 0
    usage = TokenUsageAccumulator(vocabulary_size)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        reconstruction, indices, diagnostics = model(x)
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
) -> FSQAutoencoder:
    config = {
        "levels": args.levels,
        "hidden_channels": args.hidden_channels,
        "downsample_steps": DOWNSAMPLE_STEPS,
    }
    model = FSQAutoencoder(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    vocabulary_size = model.quantizer.codebook_size
    fixed_bits = TOKENS_PER_IMAGE * math.ceil(math.log2(vocabulary_size))
    print(
        f"FSQ levels={args.levels}, vocabulary={vocabulary_size}, "
        f"fixed-length upper bound={fixed_bits} bits/image"
    )
    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        sums = torch.zeros(2, device=device)
        examples = 0
        usage = TokenUsageAccumulator(vocabulary_size)
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            reconstruction, indices, diagnostics = model(x)
            loss = F.mse_loss(reconstruction, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = [
                loss.detach(),
                diagnostics["quantization_mse"],
            ]
            sums += torch.stack(values) * x.shape[0]
            usage.update(indices)
            examples += x.shape[0]
        means = (sums / examples).tolist()
        epoch_usage = usage.statistics()
        print(
            f"tokenizer {epoch:03d}: D={means[0]:.4f}, quant={means[1]:.4f}, "
            f"PPL={epoch_usage['perplexity'].item():.1f}, "
            f"active={epoch_usage['active_codes'].item()}/{vocabulary_size}, "
            f"epoch marginal entropy="
            f"{epoch_usage['token_entropy_nats'].item() / math.log(2):.2f} bits/token"
        )
        if epoch == 1 or epoch % args.sample_every == 0:
            save_image(
                torch.cat((x[:8], reconstruction[:8])).mul(0.5).add(0.5),
                out_dir / f"tokenizer_{epoch:03d}.png",
                nrow=8,
            )
    validation = evaluate_tokenizer(model, validation_loader, device=device)
    interface_id = model_state_fingerprint(model)
    save_model_weights(
        model,
        out_dir / "tokenizer.pth",
        metadata={
            "model_name": "fsq_tokenizer",
            "interface_id": interface_id,
            "model_config": config,
            "dataset": "imagenette-128",
            "image_size": IMAGE_SIZE,
        },
    )
    print(
        f"validation tokenizer: MSE={validation['mse']:.4f}, "
        f"entropy={validation['marginal_entropy_bits_per_token']:.3f} "
        f"bits/token, active={validation['active_codes']:.0f}/"
        f"{vocabulary_size}"
    )
    return model.eval().requires_grad_(False)


def load_tokenizer(path: Path, device: torch.device) -> FSQAutoencoder:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "fsq_tokenizer":
        raise ValueError("checkpoint is not an FSQ tokenizer")
    if (
        checkpoint.get("dataset") != "imagenette-128"
        or checkpoint.get("image_size") != IMAGE_SIZE
    ):
        raise ValueError("tokenizer is not the Imagenette-128 configuration")
    model = FSQAutoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("interface_id") != model_state_fingerprint(model):
        raise ValueError(
            "tokenizer checkpoint has no valid interface identity; retrain stage 1"
        )
    return model.eval().requires_grad_(False)


def train_prior(
    args: argparse.Namespace,
    tokenizer: FSQAutoencoder,
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
                nrow=5,
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
            "model_name": "fsq_pixelcnn_prior",
            "model_config": prior_config,
            "tokenizer_interface_id": tokenizer_interface_id,
            "dataset": "imagenette-128",
            "image_size": IMAGE_SIZE,
            "conditioning": "class_conditional",
        },
    )
    print(
        f"validation prior: {validation['bits_per_token']:.3f} "
        f"bits/token, {validation['bits_per_image']:.1f} bits/image"
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vae" / "fsq"
    train_loader, validation_loader = make_imagenette_train_validation_loaders(
        args.data_dir,
        args.batch_size,
        device,
        image_size=IMAGE_SIZE,
        num_workers=args.workers,
    )
    path = out_dir / "tokenizer.pth"
    if args.stage in {"tokenizer", "all"}:
        reset_dir(str(out_dir))
        tokenizer = train_tokenizer(
            args, train_loader, validation_loader, device, out_dir
        )
    else:
        if not path.is_file():
            raise FileNotFoundError("train --stage tokenizer before the prior")
        tokenizer = load_tokenizer(path, device)
        # The tokenizer is an input to this stage; keep its handoff checkpoint.
        tokenizer_checkpoint = path.read_bytes()
        reset_dir(str(out_dir))
        path.write_bytes(tokenizer_checkpoint)
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
    parser.add_argument("--levels", type=parse_levels, default=(8, 8, 5, 5))
    parser.add_argument("--tokenizer-epochs", type=int, default=20)
    parser.add_argument("--prior-epochs", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=128)
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
