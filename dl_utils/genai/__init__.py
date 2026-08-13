"""Utilities for VAE, GAN, CycleGAN, pix2pix, and diffusion lessons.

Submodules:
    vae:      Original 256x256 VAE and diagonal-Gaussian primitives.
    vae_common: Compact Gaussian-VAE model and distribution algebra.
    vae_hierarchy: Shared two-level HVAE generator, ELBO, and diagnostics.
    quantization: VQ/FSQ quantizers and compact 32x32 tokenizers.
    token_prior: PixelCNN and causal-Transformer discrete priors.
    perceptual_autoencoder: Shared VAE-GAN/VQGAN/KL-AE network blocks.
    gan:      GAN training helpers (update_D, update_G, gradient_penalty).
    cyclegan: CycleGAN models (Generator, Discriminator, LoadData) and
              training helpers (train_epoch, weights_init).
    pix2pix:  Paired CelebA data, U-Net generator, conditional PatchGAN, and
              shared preprocessing helpers.
    sn_gan:   Conditional SN-GAN residual blocks, hinge losses, and helpers.
    sagan:    Pooled non-local attention plus conditional generator and
              projection-discriminator components.
    biggan:   Conditional BatchNorm, hierarchical latent conditioning,
              orthogonal initialization/regularization, and truncated sampling.
    stylegan_common: Equalized layers, noise control, minibatch statistics,
                     and pure PyTorch filtered resampling.
    progan:    Progressive-growing generator and critic blocks.
    stylegan:  Progressive AdaIN synthesis, W-space styles, and truncation.
    stylegan2: Modulated/demodulated synthesis, overlapping W-slot routing,
               skip RGB generation, and the residual discriminator.
    diffusion_ddpm: Discrete VP schedules and DDPM/DDIM reverse transitions.
    diffusion_score_sde: Continuous VP-SDE and probability-flow samplers.
    diffusion_unet: Compact time-conditioned convolutional backbone.
    ddpm:      Compatibility exports for the three focused diffusion modules.
"""
