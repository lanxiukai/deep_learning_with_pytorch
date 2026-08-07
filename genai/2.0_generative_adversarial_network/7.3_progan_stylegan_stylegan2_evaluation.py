"""Compare trained ProGAN, StyleGAN, and StyleGAN2 generators in one figure.

Inputs:
    output/progan/progan_generator.pth, created by 7.0_progan.py
    output/stylegan/stylegan_generator.pth, created by 7.1_stylegan.py
    output/stylegan2/stylegan2_generator.pth, created by 7.2_stylegan2.py

Output:
    output/progan_stylegan_stylegan2_evaluation/
        progan_stylegan_stylegan2_evaluation.png

The fixed-sample section sends the same z tensors to all three independently
trained models. This fixes the random conditions for comparison, but it does
not imply sample-wise semantic correspondence between their latent spaces.
"""

import torch

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.progan import ProGANGenerator
from dl_utils.genai.stylegan import StyleGANGenerator
from dl_utils.genai.stylegan2 import StyleGenerator
from dl_utils.genai.stylegan_common import denormalize
from dl_utils.plot import _backend as _  # select before importing pyplot

from matplotlib import pyplot as plt


PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "progan_stylegan_stylegan2_evaluation"
EVALUATION_PATH = OUT_DIR / "progan_stylegan_stylegan2_evaluation.png"
SEED = 42
NUM_FIXED_SAMPLES = 8
NUM_NOISE_VARIATIONS = 8
NUM_STYLE_SOURCES = 4
COARSE_MAX_RESOLUTION = 8
TRUNCATION_PSIS = (1.0, 0.7, 0.5, 0.0)

MODEL_SPECS = (
    (
        "progan",
        "ProGAN",
        ProGANGenerator,
        PROJECT_ROOT / "output" / "progan" / "progan_generator.pth",
        "7.0_progan.py",
    ),
    (
        "stylegan",
        "StyleGAN",
        StyleGANGenerator,
        PROJECT_ROOT / "output" / "stylegan" / "stylegan_generator.pth",
        "7.1_stylegan.py",
    ),
    (
        "stylegan2",
        "StyleGAN2",
        StyleGenerator,
        PROJECT_ROOT / "output" / "stylegan2" / "stylegan2_generator.pth",
        "7.2_stylegan2.py",
    ),
)


@torch.no_grad()
def estimate_missing_w_avg(generator, device, num_samples=4096):
    """Estimate W's center only when loading a pre-w_avg StyleGAN2 checkpoint."""
    total = torch.zeros(generator.style_dim, device=device)
    batch_size = 256
    for start in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - start)
        z = torch.randn(current_batch, generator.z_dim, device=device)
        total += generator.mapping(z).sum(dim=0)
    generator.w_avg.copy_(total / num_samples)


def load_generator(
    model_name,
    model_class,
    checkpoint_path,
    script_name,
    device,
):
    """Load a metadata-rich checkpoint, plus the previous StyleGAN2 format."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{model_name} checkpoint not found: {checkpoint_path}. "
            f"Run {script_name} first."
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected a metadata-rich checkpoint: {checkpoint_path}."
        )

    if "model_config" in checkpoint and "state_dict" in checkpoint:
        if checkpoint.get("model_name") != model_name:
            raise ValueError(
                f"Expected {model_name!r} at {checkpoint_path}, "
                f"found {checkpoint.get('model_name')!r}."
            )
        model_config = checkpoint["model_config"]
        state_dict = checkpoint["state_dict"]
    elif model_name == "stylegan2" and "model" in checkpoint:
        model_config = {
            "z_dim": checkpoint["z_dim"],
            "style_dim": checkpoint["style_dim"],
            "base_channels": checkpoint["base_channels"],
            "mapping_layers": checkpoint.get("mapping_layers", 4),
        }
        state_dict = checkpoint["model"]
    else:
        raise ValueError(
            f"Checkpoint metadata is incomplete: {checkpoint_path}."
        )
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint metadata is invalid: {checkpoint_path}."
        )

    generator = model_class(**model_config).to(device)
    incompatible = generator.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    legacy_missing_w_avg = model_name == "stylegan2" and missing == {"w_avg"}
    if unexpected or (missing and not legacy_missing_w_avg):
        raise ValueError(
            f"Checkpoint parameters do not match {model_name}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if legacy_missing_w_avg:
        estimate_missing_w_avg(generator, device)
        print("Estimated w_avg for a legacy StyleGAN2 checkpoint.")
    return generator.eval(), model_config


def synthesize_styles(generator, model_name, ws, noise_mode):
    """Call the generation-specific synthesis signature explicitly."""
    if model_name == "stylegan":
        return generator.synthesize(
            ws,
            resolution=32,
            alpha=1.0,
            noise_mode=noise_mode,
        )
    if model_name == "stylegan2":
        return generator.synthesize(ws, noise_mode=noise_mode)
    raise ValueError(f"{model_name} does not expose style synthesis.")


def generate_fixed_samples(generators, fixed_z):
    """Use one z batch while respecting each model's final sampling API."""
    return {
        "progan": generators["progan"](
            fixed_z,
            resolution=32,
            alpha=1.0,
        ).cpu(),
        "stylegan": generators["stylegan"](
            fixed_z,
            resolution=32,
            alpha=1.0,
            noise_mode="fixed",
        ).cpu(),
        "stylegan2": generators["stylegan2"](
            fixed_z,
            noise_mode="fixed",
        ).cpu(),
    }


def generate_noise_variations(generator, model_name, z):
    """Hold one W fixed while drawing independent layer noise per image."""
    w = generator.mapping(z)
    ws = w[:, None, :].repeat(
        NUM_NOISE_VARIATIONS,
        generator.num_ws,
        1,
    )
    return synthesize_styles(
        generator,
        model_name,
        ws,
        noise_mode="random",
    ).cpu()


def generate_style_mixing(generator, model_name, row_z, column_z):
    """Build unmixed sources and a row-coarse/column-fine mixing matrix."""
    row_w = generator.mapping(row_z)
    column_w = generator.mapping(column_z)
    row_ws = row_w[:, None, :].repeat(1, generator.num_ws, 1)
    column_ws = column_w[:, None, :].repeat(1, generator.num_ws, 1)
    row_images = synthesize_styles(
        generator,
        model_name,
        row_ws,
        noise_mode="fixed",
    )
    column_images = synthesize_styles(
        generator,
        model_name,
        column_ws,
        noise_mode="fixed",
    )
    coarse_blocks = sum(
        resolution <= COARSE_MAX_RESOLUTION
        for resolution in generator.resolutions
    )
    cutoff = coarse_blocks * generator.styles_per_block
    mixed_ws = torch.stack(
        [
            torch.cat(
                [
                    row_w[row].repeat(cutoff, 1),
                    column_w[column].repeat(
                        generator.num_ws - cutoff,
                        1,
                    ),
                ]
            )
            for row in range(NUM_STYLE_SOURCES)
            for column in range(NUM_STYLE_SOURCES)
        ]
    )
    mixed_images = synthesize_styles(
        generator,
        model_name,
        mixed_ws,
        noise_mode="fixed",
    )
    return (
        row_images.cpu(),
        column_images.cpu(),
        mixed_images.cpu(),
    )


def generate_truncation_samples(generator, model_name, z):
    """Generate one fixed latent at each requested truncation psi."""
    samples = []
    for psi in TRUNCATION_PSIS:
        if model_name == "stylegan":
            sample = generator(
                z,
                resolution=32,
                alpha=1.0,
                truncation_psi=psi,
                noise_mode="fixed",
            )
        elif model_name == "stylegan2":
            sample = generator(
                z,
                truncation_psi=psi,
                noise_mode="fixed",
            )
        else:
            raise ValueError(f"{model_name} does not support truncation.")
        samples.append(sample[0].cpu())
    return torch.stack(samples)


def display_image(axis, image):
    """Render one normalized CHW tensor without chart decorations."""
    display = denormalize(image.detach().cpu()).permute(1, 2, 0).numpy()
    axis.imshow(display)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_style_mixing_matrix(subfigure, title, mixing_data):
    """Draw one labelled source-and-mixture matrix inside a subfigure."""
    row_images, column_images, mixed_images = mixing_data
    subfigure.suptitle(title, fontsize=13)
    grid_size = NUM_STYLE_SOURCES + 1
    axes = subfigure.subplots(grid_size, grid_size, squeeze=False)
    corner = axes[0, 0]
    corner.text(
        0.5,
        0.5,
        "Coarse ↓\nFine →",
        ha="center",
        va="center",
        fontsize=9,
        transform=corner.transAxes,
    )
    corner.set_xticks([])
    corner.set_yticks([])
    for spine in corner.spines.values():
        spine.set_visible(False)

    for column, image in enumerate(column_images, start=1):
        display_image(axes[0, column], image)
        axes[0, column].set_title(f"Fine {column}", fontsize=8)
    for row, image in enumerate(row_images, start=1):
        display_image(axes[row, 0], image)
        axes[row, 0].set_ylabel(
            f"Coarse {row}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )
        for column in range(1, grid_size):
            index = (row - 1) * NUM_STYLE_SOURCES + column - 1
            display_image(axes[row, column], mixed_images[index])


def save_evaluation_figure(
    fixed_samples,
    noise_variations,
    style_mixing,
    truncation_samples,
    output_path,
):
    """Save all four comparison sections as one titled image."""
    figure = plt.figure(figsize=(15, 26), layout="constrained")
    figure.suptitle(
        "ProGAN → StyleGAN → StyleGAN2 on CIFAR-10\n"
        "Shared z fixes random conditions; independently trained models "
        "do not have sample-wise semantic correspondence.",
        fontsize=18,
        linespacing=1.35,
    )
    fixed_figure, noise_figure, mixing_figure, truncation_figure = (
        figure.subfigures(
            4,
            1,
            height_ratios=(3.5, 2.5, 7.5, 2.5),
            hspace=0.06,
        )
    )

    fixed_figure.suptitle("1. Fixed latent samples", fontsize=15)
    fixed_axes = fixed_figure.subplots(3, NUM_FIXED_SAMPLES, squeeze=False)
    for row, (_, display_name, _, _, _) in enumerate(MODEL_SPECS):
        model_name = MODEL_SPECS[row][0]
        fixed_axes[row, 0].set_ylabel(
            display_name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=11,
        )
        for axis, image in zip(
            fixed_axes[row],
            fixed_samples[model_name],
            strict=True,
        ):
            display_image(axis, image)

    noise_figure.suptitle(
        "2. Stochastic noise variations - one W per model",
        fontsize=15,
    )
    noise_axes = noise_figure.subplots(
        2,
        NUM_NOISE_VARIATIONS,
        squeeze=False,
    )
    for row, model_name in enumerate(("stylegan", "stylegan2")):
        label = "StyleGAN" if model_name == "stylegan" else "StyleGAN2"
        noise_axes[row, 0].set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=11,
        )
        for column, (axis, image) in enumerate(
            zip(
                noise_axes[row],
                noise_variations[model_name],
                strict=True,
            ),
            start=1,
        ):
            display_image(axis, image)
            if row == 0:
                axis.set_title(f"Noise {column}", fontsize=8)

    mixing_figure.suptitle(
        "3. Style mixing - 4-8 px styles from rows, 16-32 px styles from columns",
        fontsize=15,
    )
    mixing_subfigures = mixing_figure.subfigures(1, 2, wspace=0.04)
    for subfigure, model_name, title in zip(
        mixing_subfigures,
        ("stylegan", "stylegan2"),
        ("StyleGAN (AdaIN)", "StyleGAN2 (modulation/demodulation)"),
        strict=True,
    ):
        draw_style_mixing_matrix(
            subfigure,
            title,
            style_mixing[model_name],
        )

    truncation_figure.suptitle(
        "4. Truncation comparison - one fixed z per model",
        fontsize=15,
    )
    truncation_axes = truncation_figure.subplots(
        2,
        len(TRUNCATION_PSIS),
        squeeze=False,
    )
    for row, model_name in enumerate(("stylegan", "stylegan2")):
        label = "StyleGAN" if model_name == "stylegan" else "StyleGAN2"
        truncation_axes[row, 0].set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=11,
        )
        for axis, image, psi in zip(
            truncation_axes[row],
            truncation_samples[model_name],
            TRUNCATION_PSIS,
            strict=True,
        ):
            display_image(axis, image)
            if row == 0:
                axis.set_title(f"psi={psi:.1f}", fontsize=9)

    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    set_seed(SEED)
    device = try_gpu()
    generators = {}
    configurations = {}
    for model_name, _, model_class, path, script_name in MODEL_SPECS:
        generator, model_config = load_generator(
            model_name,
            model_class,
            path,
            script_name,
            device,
        )
        generators[model_name] = generator
        configurations[model_name] = model_config

    latent_dimensions = {
        int(config["z_dim"]) for config in configurations.values()
    }
    if len(latent_dimensions) != 1:
        raise ValueError(
            "The three checkpoints need the same z_dim for shared-z sampling."
        )
    z_dim = latent_dimensions.pop()
    reset_dir(str(OUT_DIR))

    with torch.inference_mode():
        torch.manual_seed(SEED)
        fixed_z = torch.randn(NUM_FIXED_SAMPLES, z_dim, device=device)
        fixed_samples = generate_fixed_samples(generators, fixed_z)

        torch.manual_seed(SEED + 1)
        noise_z = torch.randn(1, z_dim, device=device)
        noise_variations = {
            model_name: generate_noise_variations(
                generators[model_name],
                model_name,
                noise_z,
            )
            for model_name in ("stylegan", "stylegan2")
        }

        torch.manual_seed(SEED + 2)
        row_z = torch.randn(NUM_STYLE_SOURCES, z_dim, device=device)
        column_z = torch.randn(NUM_STYLE_SOURCES, z_dim, device=device)
        style_mixing = {
            model_name: generate_style_mixing(
                generators[model_name],
                model_name,
                row_z,
                column_z,
            )
            for model_name in ("stylegan", "stylegan2")
        }

        torch.manual_seed(SEED + 3)
        truncation_z = torch.randn(1, z_dim, device=device)
        truncation_samples = {
            model_name: generate_truncation_samples(
                generators[model_name],
                model_name,
                truncation_z,
            )
            for model_name in ("stylegan", "stylegan2")
        }

    save_evaluation_figure(
        fixed_samples,
        noise_variations,
        style_mixing,
        truncation_samples,
        EVALUATION_PATH,
    )


if __name__ == "__main__":
    main()
