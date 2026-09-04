"""Evaluate cVAE generation paths that the training loss cannot validate.

This script trains a small held-out MNIST classifier only as an evaluator,
then measures:

* class compliance of samples drawn from p(z | c);
* within-class feature diversity relative to real test images;
* posterior-mean reconstruction and a decoder condition-shuffle intervention;
* the standard-prior main model, plus learned-prior and deterministic
  controls when their final artifacts are available.

The classifier is not part of the cVAE checkpoint or generation path.

Data:
    data/mnist, downloaded automatically by torchvision when absent. Digit
    labels train the evaluator and condition the compared generators.

Checkpoints:
    output/vae/conditional_vae/standard-prior/model.pth: required main model
    output/vae/conditional_vae/learned-prior/model.pth: optional control
    output/vae/conditional_vae/deterministic/model.pth: optional control

Outputs:
    output/vae/conditional_vae/evaluation/metrics.json: complete comparison
    output/vae/conditional_vae/evaluation/<variant>_conditional_samples.png

Evaluation data -- MNIST:
Classifier training images:        60,000
Classifier samples per epoch:      59,904 (234 full batches)
Classifier epochs:                      3
Classifier optimizer updates:         702
Test images:                       10,000
Batch size:                           256
Generated samples per model:         1,280 (128 per class)
Real diversity reference:            1,280 (128 per class)
Maximum reconstruction examples:     5,000 per VAE variant

Default dimensions:
Evaluation input:                  32x32 grayscale
Generated image:                   32x32 grayscale
Latent vector:                         16 values (VAE variants)

Model size:
Standard-prior CVAE:                0.971 M parameters (required)
Learned-prior CVAE:                 0.980 M parameters (optional)
Deterministic baseline:             0.235 M parameters (optional)
Evaluator classifier:               0.056 M parameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.conditional import (
    ConditionalMeanDecoder32,
    ConditionalVAE32,
)
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar

PROJECT_ROOT = infer_project_root()
DEFAULT_ROOT = PROJECT_ROOT / "output" / "vae" / "conditional_vae"


class DigitClassifier32(nn.Module):
    """Small task network used only to audit generated-condition compliance."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, 10)

    def encode(self, x: Tensor) -> Tensor:
        return self.features(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encode(x))


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
    test_set = datasets.MNIST(
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
        DataLoader(test_set, shuffle=False, drop_last=False, **common),
    )


def train_classifier(
    model: DigitClassifier32,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    device: torch.device,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        examples = 0
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(dim=1) == labels).sum())
            examples += x.shape[0]
        print(
            f"classifier epoch {epoch:02d}: "
            f"train accuracy={correct / examples:.4f}"
        )

    model.eval()
    correct = 0
    examples = 0
    with torch.inference_mode():
        for x, labels in test_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            correct += int((model(x).argmax(dim=1) == labels).sum())
            examples += x.shape[0]
    return correct / examples


def _load_model(
    path: Path, device: torch.device
) -> ConditionalVAE32 | ConditionalMeanDecoder32:
    checkpoint = torch.load(
        path, map_location=device, weights_only=True
    )
    model_name = checkpoint.get("model_name")
    config = checkpoint["model_config"]
    if model_name == "conditional_vae":
        model: ConditionalVAE32 | ConditionalMeanDecoder32 = (
            ConditionalVAE32(**config)
        )
    elif model_name == "conditional_mean_decoder":
        model = ConditionalMeanDecoder32(**config)
    else:
        raise ValueError(f"{path} is not a supported conditional checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def _mean_pairwise_distance(features: Tensor) -> float:
    if features.shape[0] < 2:
        return 0.0
    return float(torch.pdist(features.float(), p=2).mean())


@torch.inference_mode()
def generation_metrics(
    model: ConditionalVAE32 | ConditionalMeanDecoder32,
    classifier: DigitClassifier32,
    *,
    samples_per_class: int,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    labels = torch.arange(10, device=device).repeat_interleave(
        samples_per_class
    )
    if isinstance(model, ConditionalVAE32):
        images = model.generate(labels)
    else:
        images = model.generate(labels)
    features = classifier.encode(images)
    probabilities = classifier(images).softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    per_class_compliance = []
    per_class_diversity = []
    for class_index in range(10):
        mask = labels == class_index
        per_class_compliance.append(
            float((predictions[mask] == class_index).float().mean())
        )
        per_class_diversity.append(
            _mean_pairwise_distance(features[mask])
        )
    return {
        "condition_compliance": float(
            (predictions == labels).float().mean()
        ),
        "target_probability": float(
            probabilities.gather(1, labels[:, None]).mean()
        ),
        "feature_diversity": sum(per_class_diversity) / 10,
        "per_class_compliance": per_class_compliance,
        "per_class_feature_diversity": per_class_diversity,
    }, images


@torch.inference_mode()
def real_diversity_reference(
    loader: DataLoader,
    classifier: DigitClassifier32,
    *,
    samples_per_class: int,
    device: torch.device,
) -> dict[str, object]:
    buckets: list[list[Tensor]] = [[] for _ in range(10)]
    for x, labels in loader:
        for image, label in zip(x, labels):
            bucket = buckets[int(label)]
            if len(bucket) < samples_per_class:
                bucket.append(image)
        if all(len(bucket) == samples_per_class for bucket in buckets):
            break
    per_class = []
    for bucket in buckets:
        images = torch.stack(bucket).to(device)
        per_class.append(_mean_pairwise_distance(classifier.encode(images)))
    return {
        "feature_diversity": sum(per_class) / 10,
        "per_class_feature_diversity": per_class,
    }


@torch.inference_mode()
def posterior_and_shuffle_metrics(
    model: ConditionalVAE32,
    loader: DataLoader,
    *,
    device: torch.device,
    max_examples: int,
) -> dict[str, float]:
    correct_distortion = 0.0
    shuffled_distortion = 0.0
    rate = 0.0
    posterior_prior_mean_gap = 0.0
    examples = 0
    for x, labels in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        labels = labels[:remaining].to(device, non_blocking=True)
        q_mu, q_logvar = model.encode(x, labels)
        p_mu, p_logvar = model.prior(labels)
        correct = model.decode(q_mu, labels)
        shuffled_labels = (labels + 1) % model.num_classes
        shuffled = model.decode(q_mu, shuffled_labels)
        correct_distortion += float(
            F.binary_cross_entropy(correct, x, reduction="sum")
        )
        shuffled_distortion += float(
            F.binary_cross_entropy(shuffled, x, reduction="sum")
        )
        rate += float(
            diagonal_gaussian_kl_from_logvar(
                q_mu, q_logvar, p_mu, p_logvar
            ).sum()
        )
        posterior_prior_mean_gap += float(
            (q_mu - p_mu).square().sum()
        )
        examples += x.shape[0]
    return {
        "posterior_mean_distortion": correct_distortion / examples,
        "shuffled_decoder_condition_distortion": (
            shuffled_distortion / examples
        ),
        "condition_shuffle_distortion_ratio": (
            shuffled_distortion / correct_distortion
        ),
        "conditional_rate": rate / examples,
        "posterior_prior_mean_squared_gap": (
            posterior_prior_mean_gap / examples
        ),
    }


def _checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "standard-prior": args.standard_prior_checkpoint,
        "learned-prior": args.learned_prior_checkpoint,
        "deterministic": args.deterministic_checkpoint,
    }


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args, device)
    classifier = DigitClassifier32().to(device)
    classifier_accuracy = train_classifier(
        classifier,
        train_loader,
        test_loader,
        epochs=args.classifier_epochs,
        device=device,
    )
    if classifier_accuracy < 0.97:
        print(
            "warning: evaluator accuracy is below 97%; treat compliance "
            "numbers as unreliable"
        )

    paths = _checkpoint_paths(args)
    if not paths["standard-prior"].exists():
        raise FileNotFoundError(
            f"missing {paths['standard-prior']}; run 3.0_conditional_vae.py"
        )
    missing_controls = [
        name
        for name in ("learned-prior", "deterministic")
        if not paths[name].exists()
    ]
    if missing_controls and args.require_controls:
        raise FileNotFoundError(
            "missing required control checkpoints: "
            + ", ".join(missing_controls)
        )

    out_dir = DEFAULT_ROOT / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "classifier_test_accuracy": classifier_accuracy,
        "real_reference": real_diversity_reference(
            test_loader,
            classifier,
            samples_per_class=args.samples_per_class,
            device=device,
        ),
        "models": {},
        "missing_optional_controls": missing_controls,
    }
    for name, path in paths.items():
        if not path.exists():
            print(f"skip {name}: checkpoint not found at {path}")
            continue
        model = _load_model(path, device)
        metrics, images = generation_metrics(
            model,
            classifier,
            samples_per_class=args.samples_per_class,
            device=device,
        )
        if isinstance(model, ConditionalVAE32):
            metrics.update(
                posterior_and_shuffle_metrics(
                    model,
                    test_loader,
                    device=device,
                    max_examples=args.max_reconstruction_examples,
                )
            )
        results["models"][name] = metrics
        display_count = min(args.samples_per_class, 8)
        display = images.reshape(
            10, args.samples_per_class, 1, 32, 32
        )[:, :display_count].flatten(0, 1)
        save_image(
            display,
            out_dir / f"{name}_conditional_samples.png",
            nrow=display_count,
        )
        print(
            f"{name}: compliance={metrics['condition_compliance']:.4f}, "
            f"feature diversity={metrics['feature_diversity']:.4f}"
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved evaluation to {out_dir}")


def smoke_test() -> None:
    set_seed(7)
    device = torch.device("cpu")
    classifier = DigitClassifier32().to(device)
    model = ConditionalVAE32(
        latent_dim=4,
        condition_dim=8,
        hidden_channels=32,
        learned_prior=True,
    ).to(device)
    metrics, images = generation_metrics(
        model,
        classifier,
        samples_per_class=3,
        device=device,
    )
    loader = DataLoader(
        list(
            zip(
                torch.rand(20, 1, 32, 32),
                torch.arange(20) % 10,
            )
        ),
        batch_size=5,
    )
    metrics.update(
        posterior_and_shuffle_metrics(
            model, loader, device=device, max_examples=20
        )
    )
    assert images.shape == (30, 1, 32, 32)
    assert all(
        torch.isfinite(torch.tensor(value))
        for key, value in metrics.items()
        if not key.startswith("per_class")
    )
    print("smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--classifier-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples-per-class", type=int, default=128)
    parser.add_argument(
        "--max-reconstruction-examples", type=int, default=5_000
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--learned-prior-checkpoint",
        type=Path,
        default=DEFAULT_ROOT / "learned-prior" / "model.pth",
    )
    parser.add_argument(
        "--standard-prior-checkpoint",
        type=Path,
        default=DEFAULT_ROOT / "standard-prior" / "model.pth",
    )
    parser.add_argument(
        "--deterministic-checkpoint",
        type=Path,
        default=DEFAULT_ROOT / "deterministic" / "model.pth",
    )
    parser.add_argument("--require-controls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
