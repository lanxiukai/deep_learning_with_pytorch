"""
Evaluate a previously trained Conditional GAN (cGAN) checkpoint — sample
faces with/without glasses, then interpolate in latent (z) and label space.

Requires output/gan/cgan/cgan.pth — run 4.0_conditional_gan.py first.

Outputs:
    output/gan/cgan/evaluation/, reset at the start of every run.
"""

import torch

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.gan import Generator
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / 'output' / 'gan' / 'cgan' / 'evaluation'

checkpoint_path = OUT_DIR.parent / 'cgan.pth'
if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found at {checkpoint_path}. "
        "Run 4.0_conditional_gan.py first."
    )

def main():
    """Load a trained cGAN checkpoint and run the full evaluation pipeline."""
    # ensure a clean output directory before each run
    reset_dir(str(OUT_DIR))

    # determine the device automatically
    device = try_gpu()

    z_dim, img_channels, features = 100, 3, 16

    torch.manual_seed(0)

    generator = Generator(z_dim + 2, img_channels, features).to(device)
    generator.load_state_dict(torch.load(checkpoint_path,
                                         map_location=device))
    generator.eval()

    def _savefig(name):
        """Save current figure to OUT_DIR and close it."""
        path = OUT_DIR / f'{name}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')

    # ── Helper definitions ──────────────────────────────────────────
    # One‑hot label constants used throughout the evaluation.
    label_g  = torch.zeros(2, 1, 1); label_g[0, :, :]  = 1  # glasses
    label_ng = torch.zeros(2, 1, 1); label_ng[1, :, :] = 1  # no glasses

    def _denorm(img):
        """(C,H,W) tensor in [-1,1] → (H,W,C) numpy array in [0,1]."""
        return (img / 2 + 0.5).permute(1, 2, 0).numpy()

    def _show(img):
        """Display a single denormalised image tensor."""
        plt.imshow(_denorm(img))
        plt.xticks([])
        plt.yticks([])

    def _gen(noise, label):
        """Concatenate noise + label, run generator, return CPU tensor."""
        x = torch.cat([noise, label], dim=1).to(device)
        return generator(x).cpu().detach()

    def _plot_grid(images, rows, cols, name, figsize=None):
        """Plot (N,C,H,W) batch of images in a rows×cols grid."""
        if figsize is None:
            figsize = (cols * 2.5, rows * 2.5)
        plt.figure(figsize=figsize, dpi=300)
        for i in range(len(images)):
            plt.subplot(rows, cols, i + 1)
            _show(images[i])
        plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
        _savefig(name)

    def _sample_grid(n, active_channel, name, rows=4, cols=8):
        """Generate `n` images with a one‑hot label, plot grid, return noise."""
        noise = torch.randn(n, z_dim, 1, 1)
        labels = torch.zeros(n, 2, 1, 1)
        labels[:, active_channel, :, :] = 1
        fake = _gen(noise, labels)
        _plot_grid(fake, rows, cols, name)
        return noise

    def _interp_1d(z_a, z_b, label_a, label_b, weights, name):
        """Plot 1×N row: interpolate z (a→b) AND label (a→b) together."""
        plt.figure(figsize=(len(weights) * 4, 4), dpi=300)
        for i, w in enumerate(weights):
            plt.subplot(1, len(weights), i + 1)
            z = w * z_b + (1 - w) * z_a
            label = w * label_b + (1 - w) * label_a
            fake = _gen(z.reshape(1, -1, 1, 1), label.reshape(1, 2, 1, 1))
            _show(fake[0])
        plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
        _savefig(name)

    def _combo_2x2(z0, z1, label0, label1, name):
        """Plot 2×2: every combination of {z0, z1} × {label0, label1}."""
        plt.figure(figsize=(20, 5), dpi=300)
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            p, q = i // 2, i % 2
            z = z1 * p + z0 * (1 - p)
            label = label1 * q + label0 * (1 - q)
            fake = _gen(z.reshape(1, -1, 1, 1), label.reshape(1, 2, 1, 1))
            _show(fake[0])
        plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
        _savefig(name)

    def _interp_grid_6x6(z_a, z_b, label_a, label_b, name):
        """Plot 6×6 grid: jointly interpolate z (a→b) and label (a→b)."""
        plt.figure(figsize=(20, 20), dpi=300)
        for i in range(36):
            plt.subplot(6, 6, i + 1)
            p, q = i // 6, i % 6
            z = z_b * (p / 5) + z_a * (1 - p / 5)
            label = label_b * (q / 5) + label_a * (1 - q / 5)
            fake = _gen(z.reshape(1, -1, 1, 1), label.reshape(1, 2, 1, 1))
            _show(fake[0])
        plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
        _savefig(name)

    # ── 1. Sample faces with glasses / no‑glasses ──────────────────

    noise_g  = _sample_grid(32, active_channel=0, name='01_sample_glasses')  # glasses
    noise_ng = _sample_grid(32, active_channel=1, name='02_sample_no_glasses')  # no glasses

    # Pick representative z vectors (indices from original notebook).
    z_male_g, z_female_g   = noise_g[0],  noise_g[20]
    z_male_ng, z_female_ng = noise_ng[8], noise_ng[12]

    weights = [0, 0.25, 0.5, 0.75, 1]

    # ── 2. Label interpolation (fix z, glasses → no glasses) ───────

    _interp_1d(z_female_g, z_female_g, label_g, label_ng, weights, '03_label_interp_female')  # fix z (female), vary label
    _interp_1d(z_male_g,   z_male_g,   label_g, label_ng, weights, '04_label_interp_male')    # fix z (male),   vary label — Exercise 5.1

    # ── 3. Latent interpolation (fix label, male → female) ──────────

    _interp_1d(z_male_ng, z_female_ng, label_ng, label_ng, weights, '05_latent_interp_no_glasses')  # fix label (no glasses), vary z
    _interp_1d(z_male_ng, z_female_ng, label_g,  label_g,  weights, '06_latent_interp_glasses')    # fix label (glasses),    vary z — Exercise 5.2

    # ── 4. 2×2 combinations (vary z × label) ────────────────────────

    _combo_2x2(z_male_g, z_female_g, label_g, label_ng, '07_combo_glasses_z')       # z sampled w/ glasses prompt
    _combo_2x2(z_male_ng, z_female_ng, label_g, label_ng, '08_combo_no_glasses_z')  # z sampled w/ no‑glasses prompt — Exercise 5.3

    # ── 5. 6×6 interpolation grid (vary z + label jointly) ──────────

    _interp_grid_6x6(z_male_ng, z_female_ng, label_g, label_ng, '09_interp_grid_6x6')


if __name__ == '__main__':
    main()
