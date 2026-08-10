"""Qualitatively evaluate the ProGAN to StyleGAN2 lesson sequence.

The evaluation separates the mechanisms introduced across the three lessons:
    - all generators receive shared fixed z samples;
    - StyleGAN and StyleGAN2 vary stochastic noise while holding W fixed;
    - their coarse and fine styles are mixed explicitly;
    - their learned ``w_avg`` buffers drive truncation comparisons.

Shared inputs fix random conditions only.  The independently trained latent
spaces do not have sample-wise semantic correspondence.  This compact lesson
uses current metadata-rich checkpoints and intentionally omits legacy formats,
FID reproduction, and a large evaluation framework.

Inputs:
    output/progan/progan_generator.pth, created by 7.0_progan.py
    output/stylegan/stylegan_generator.pth, created by 7.1_stylegan.py
    output/stylegan2/stylegan2_generator.pth, created by 7.2_stylegan2.py

Outputs:
    output/progan_stylegan_stylegan2_evaluation/fixed_samples.png
    output/progan_stylegan_stylegan2_evaluation/noise_variations.png
    output/progan_stylegan_stylegan2_evaluation/stylegan_style_mixing.png
    output/progan_stylegan_stylegan2_evaluation/stylegan2_style_mixing.png
    output/progan_stylegan_stylegan2_evaluation/truncation.png
"""

import torch

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.progan import ProGANGenerator
from dl_utils.genai.stylegan import StyleGANGenerator
from dl_utils.genai.stylegan2 import StyleGenerator
from dl_utils.plot.images import save_image_row_grid


PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "progan_stylegan_stylegan2_evaluation"

SEED = 42
NUM_FIXED_SAMPLES = 8
NUM_NOISE_VARIATIONS = 8
NUM_STYLE_SOURCES = 4
COARSE_MAX_RESOLUTION = 8
TRUNCATION_PSIS = (1.0, 0.7, 0.5, 0.0)

MODEL_SPECS = (
    (
        "progan",
        "ProGAN EMA",
        ProGANGenerator,
        PROJECT_ROOT / "output" / "progan" / "progan_generator.pth",
        "7.0_progan.py",
        5,
    ),
    (
        "stylegan",
        "StyleGAN EMA",
        StyleGANGenerator,
        PROJECT_ROOT / "output" / "stylegan" / "stylegan_generator.pth",
        "7.1_stylegan.py",
        6,
    ),
    (
        "stylegan2",
        "StyleGAN2 EMA",
        StyleGenerator,
        PROJECT_ROOT / "output" / "stylegan2" / "stylegan2_generator.pth",
        "7.2_stylegan2.py",
        6,
    ),
)


def load_generator(
    model_name,
    model_class,
    checkpoint_path,
    script_name,
    expected_format_version,
    device,
):
    """Load one EMA generator produced by the current lesson scripts."""
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

    expected_metadata = {
        "model_name": model_name,
        "format_version": expected_format_version,
        "conditioning": "unconditional",
        "weights": "ema",
    }
    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"{checkpoint_path} has {key}={checkpoint.get(key)!r}; "
                f"expected {expected!r}. Retrain it with {script_name}."
            )

    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint metadata is incomplete: {checkpoint_path}."
        )

    generator = model_class(**model_config).to(device)
    generator.load_state_dict(state_dict, strict=True)
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


@torch.inference_mode()
def generate_fixed_samples(generators, fixed_z):
    """Generate deterministic final-resolution samples from one shared z."""
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


@torch.inference_mode()
def generate_noise_variations(generator, model_name, z):
    """Hold one W fixed while drawing independent per-layer noise."""
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


@torch.inference_mode()
def generate_style_mixing(generator, model_name, row_z, column_z):
    """Combine low-resolution row styles with high-resolution column styles."""
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

    cutoff = generator.num_ws_for_resolution(COARSE_MAX_RESOLUTION)
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
    mixed_images = mixed_images.reshape(
        NUM_STYLE_SOURCES,
        NUM_STYLE_SOURCES,
        *mixed_images.shape[1:],
    )
    return row_images.cpu(), column_images.cpu(), mixed_images.cpu()


@torch.inference_mode()
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


def save_fixed_samples(fixed_samples):
    """Save the shared-z comparison for all three generator families."""
    names = ("progan", "stylegan", "stylegan2")
    save_image_row_grid(
        [fixed_samples[name] for name in names],
        ["ProGAN EMA", "StyleGAN EMA", "StyleGAN2 EMA"],
        OUT_DIR / "fixed_samples.png",
        title=(
            "Shared-z samples from independently trained generators "
            "(no semantic alignment implied)"
        ),
        column_labels=[
            f"Shared z {index + 1}" for index in range(NUM_FIXED_SAMPLES)
        ],
    )


def save_noise_variations(noise_variations):
    """Save stochastic-noise rows while W remains unchanged."""
    save_image_row_grid(
        [
            noise_variations["stylegan"],
            noise_variations["stylegan2"],
        ],
        ["StyleGAN EMA", "StyleGAN2 EMA"],
        OUT_DIR / "noise_variations.png",
        title="One fixed W with independent stochastic layer noise",
        column_labels=[
            f"Noise {index + 1}" for index in range(NUM_NOISE_VARIATIONS)
        ],
    )


def save_style_mixing(model_name, display_name, mixing_data):
    """Save one source-labelled coarse/fine style-mixing matrix."""
    row_images, column_images, mixed_images = mixing_data
    corner = torch.zeros_like(row_images[:1])
    image_rows = [torch.cat([corner, column_images], dim=0)]
    image_rows.extend(
        torch.cat([row_images[row : row + 1], mixed_images[row]], dim=0)
        for row in range(NUM_STYLE_SOURCES)
    )
    save_image_row_grid(
        image_rows,
        ["Fine sources"] + [
            f"Coarse {index + 1}" for index in range(NUM_STYLE_SOURCES)
        ],
        OUT_DIR / f"{model_name}_style_mixing.png",
        title=(
            f"{display_name}: 4-8 px styles from rows, "
            "16-32 px styles from columns"
        ),
        column_labels=["Coarse source"] + [
            f"Fine {index + 1}" for index in range(NUM_STYLE_SOURCES)
        ],
    )


def save_truncation(truncation_samples):
    """Save learned-W-center interpolation for both style generators."""
    save_image_row_grid(
        [
            truncation_samples["stylegan"],
            truncation_samples["stylegan2"],
        ],
        ["StyleGAN EMA", "StyleGAN2 EMA"],
        OUT_DIR / "truncation.png",
        title="One fixed z interpolated toward each model's learned w_avg",
        column_labels=[f"psi={psi:.1f}" for psi in TRUNCATION_PSIS],
    )


def main():
    set_seed(SEED)
    device = try_gpu()

    generators = {}
    configurations = {}
    for (
        model_name,
        _,
        model_class,
        path,
        script_name,
        format_version,
    ) in MODEL_SPECS:
        generator, model_config = load_generator(
            model_name,
            model_class,
            path,
            script_name,
            format_version,
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

    save_fixed_samples(fixed_samples)
    save_noise_variations(noise_variations)
    save_style_mixing(
        "stylegan",
        "StyleGAN EMA (AdaIN)",
        style_mixing["stylegan"],
    )
    save_style_mixing(
        "stylegan2",
        "StyleGAN2 EMA (modulation/demodulation)",
        style_mixing["stylegan2"],
    )
    save_truncation(truncation_samples)


if __name__ == "__main__":
    main()
