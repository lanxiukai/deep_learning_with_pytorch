"""Utilities for VAE, GAN, CycleGAN, pix2pix, and DDPM examples.

Submodules:
    vae:      VAE model (VAEEncoder, VAEDecoder, VAE) and training constants.
    gan:      GAN training helpers (update_D, update_G, gradient_penalty).
    cyclegan: CycleGAN models (Generator, Discriminator, LoadData) and
              training helpers (train_epoch, weights_init).
    pix2pix:  Paired CelebA data, U-Net generator, conditional PatchGAN, and
              shared preprocessing helpers.
"""
