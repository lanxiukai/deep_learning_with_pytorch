"""Utilities for VAE, GAN, CycleGAN, pix2pix, modern GANs, and DDPM.

Submodules:
    vae:      VAE model (VAEEncoder, VAEDecoder, VAE) and training constants.
    gan:      GAN training helpers (update_D, update_G, gradient_penalty).
    cyclegan: CycleGAN models (Generator, Discriminator, LoadData) and
              training helpers (train_epoch, weights_init).
    pix2pix:  Paired CelebA data, U-Net generator, conditional PatchGAN, and
              shared preprocessing helpers.
    sn_gan:   Conditional SN-GAN residual blocks, hinge losses, and helpers.
    sagan:    Self-attention extensions for the SN-GAN generator and
              discriminator.
    biggan:   Conditional BatchNorm, hierarchical latent conditioning,
              orthogonal regularization, and truncated sampling.
    stylegan2: Mapping/synthesis blocks, modulated convolutions, stochastic
               noise, and the residual discriminator.
"""
