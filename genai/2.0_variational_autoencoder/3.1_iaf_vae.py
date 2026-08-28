"""IAF-VAE: expand only the variational posterior with an exact flow density.

The encoder first samples z0 from a diagonal Gaussian.  Each affine inverse
autoregressive layer then computes

    z_t = shift(z_{t-1,<i}, h(x)) + exp(log_scale) * z_{t-1}

with a triangular Jacobian.  The posterior density subtracts every layer's
log-determinant.  The decoder consumes z_T, while unconditional generation
still samples z directly from N(0, I) and never runs the posterior flow.
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import reproducibility_metadata
from dl_utils.vae.inference import (
    AffineIAFLayer,
    FlowGaussianVAE32,
    importance_diagnostics,
    importance_log_weights,
    log_mean_exp,
    model_config,
)


PROJECT_ROOT = infer_project_root()


def make_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    train_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=True,
        download=True,
        transform=transform,
    )
    validation_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=False,
        download=True,
        transform=transform,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(validation_set, shuffle=False, drop_last=False, **common),
    )


@torch.inference_mode()
def evaluate(
    model: FlowGaussianVAE32,
    loader: DataLoader,
    *,
    max_examples: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        log_weights, terms = importance_log_weights(
            model, x, particles=1
        )
        diagnostics = importance_diagnostics(log_weights, terms)
        for name, value in diagnostics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * x.shape[0]
        examples += x.shape[0]
    return {name: value / examples for name, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders(args, device)
    model = FlowGaussianVAE32(
        latent_dim=args.latent_dim,
        hidden_channels=args.hidden_channels,
        context_dim=args.context_dim,
        flow_layers=args.flow_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        max_abs_log_scale=args.max_abs_log_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            log_weights, terms = importance_log_weights(
                model, x, particles=1
            )
            loss = -log_mean_exp(log_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            diagnostics = importance_diagnostics(log_weights, terms)
            diagnostics["loss"] = loss.detach()
            for name, value in diagnostics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * x.shape[0]
            examples += x.shape[0]
        means = {name: value / examples for name, value in totals.items()}
        print(
            f"epoch {epoch:03d}: loss={means['loss']:.3f}, "
            f"ELBO={means['bound']:.3f}, "
            f"log-det={means['log_determinant']:.3f}, "
            f"scale=[{means['log_scale_min']:.3f}, "
            f"{means['log_scale_max']:.3f}]"
        )

    validation = evaluate(
        model,
        validation_loader,
        max_examples=args.validation_examples,
        device=device,
    )
    out_dir = PROJECT_ROOT / "output" / "vae" / "iaf_vae"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": "iaf_vae",
            "roadmap_role": "focused_branch",
            "roadmap_step": "2X",
            "direct_baseline": "diagonal-posterior standard VAE",
            "visible_increment": "short inverse autoregressive flow posterior",
            "posterior_family": model.posterior_family,
            "state_dict": model.state_dict(),
            "model_config": model_config(model),
            "training_particles": 1,
            "training_updates": args.epochs * len(train_loader),
            "observation": "independent Bernoulli mean",
            "image_size": 32,
            "validation_metrics": validation,
            **reproducibility_metadata(
                models={"iaf_vae": model},
                seed=args.seed,
                training_budget={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.epochs * len(train_loader),
                    "particles_per_example": 1,
                },
                data_preprocessing={
                    "resize": [32, 32],
                    "value_range": [0.0, 1.0],
                    "augmentation": "none",
                },
            ),
        },
        out_dir / "model.pth",
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(64, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=8)
    print(
        f"validation ELBO={validation['bound']:.3f}, "
        f"log-det={validation['log_determinant']:.3f}; saved {out_dir}"
    )


def _check_triangular_jacobian() -> None:
    layer = AffineIAFLayer(
        latent_dim=4,
        context_dim=3,
        hidden_dim=12,
        max_abs_log_scale=1.5,
    ).double()
    torch.nn.init.normal_(
        layer.autoregressive.output.weight, std=0.2
    )
    torch.nn.init.normal_(
        layer.autoregressive.output.bias, std=0.1
    )
    context = torch.randn(1, 1, 3, dtype=torch.double)
    z = torch.randn(4, dtype=torch.double, requires_grad=True)

    def transform(vector: Tensor) -> Tensor:
        output, _, _ = layer(
            vector.reshape(1, 1, 4), context
        )
        return output.reshape(4)

    jacobian = torch.autograd.functional.jacobian(transform, z)
    output, log_determinant, _ = layer(
        z.reshape(1, 1, 4), context
    )
    del output
    assert torch.allclose(
        torch.triu(jacobian, diagonal=1),
        torch.zeros_like(jacobian),
        atol=1e-10,
    )
    expected = jacobian.diagonal().abs().log().sum()
    assert torch.allclose(
        expected, log_determinant.squeeze(), atol=1e-8
    )


def smoke_test() -> None:
    set_seed(7)
    _check_triangular_jacobian()
    model = FlowGaussianVAE32(
        latent_dim=6,
        hidden_channels=32,
        context_dim=16,
        flow_layers=2,
        flow_hidden_dim=24,
        max_abs_log_scale=1.5,
    )
    x = torch.rand(4, 1, 32, 32)
    log_weights, terms = importance_log_weights(
        model, x, particles=3, particle_chunk_size=2
    )
    loss = -log_mean_exp(log_weights).mean()
    loss.backward()
    assert log_weights.shape == (4, 3)
    assert terms["log_determinant"].shape == (4, 3)
    assert model.flows[0].autoregressive.output.weight.grad is not None
    assert model.sample(2, device=torch.device("cpu")).shape == (
        2,
        1,
        32,
        32,
    )
    assert torch.isfinite(loss)
    print(
        "smoke test passed: triangular Jacobian, exact log-det, "
        f"loss={loss.item():.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--flow-layers", type=int, default=2)
    parser.add_argument("--flow-hidden-dim", type=int, default=128)
    parser.add_argument("--max-abs-log-scale", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--validation-examples", type=int, default=2_048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
