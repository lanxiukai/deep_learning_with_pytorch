"""Utilities for VAE, GAN, CycleGAN, pix2pix, modern GANs, and DDPM.

Submodules:
    vae:      VAE model (VAEEncoder, VAEDecoder, VAE) and training constants.
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
"""
