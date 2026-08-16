"""Shared GAN training helpers — discriminator and generator update steps,
the WGAN-GP gradient penalty, and cGAN model architectures (Generator, Critic)."""

import torch
import torch.nn as nn


def gradient_penalty(net_D, real_X, fake_X, device, lambda_gp=10):
    """λ · E[(‖∇_x̂ D(x̂)‖₂ − 1)²] on random interpolations x̂ = ε·real + (1−ε)·fake.

    A 1-Lipschitz function has gradients of norm at most 1 everywhere;
    penalizing deviations from 1 along the straight lines between real and
    fake samples is a soft, differentiable way to enforce it.
    """
    batch_size = real_X.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = (epsilon * real_X + (1 - epsilon) * fake_X).requires_grad_(True)
    d_hat = net_D(x_hat)
    # create_graph=True: the gradient of the penalty itself must backprop
    # into the critic, so the first-order graph is kept.
    gradients = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True, retain_graph=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    return lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def update_D(X, Z, net_D, net_G, loss, trainer_D, real_label: float = 1.0):
    """Update discriminator."""
    batch_size = X.shape[0]
    # Prepare the labels
    real_labels = torch.full((batch_size,), real_label, device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)

    trainer_D.zero_grad()
    real_Y = net_D(X)  # The discriminator results of the realistic samples
    fake_X = net_G(Z)  # Generated (fake) samples from noise Z
    # .detach() skips backprop through net_G here — those gradients would
    # be wasted anyway, since trainer_D only updates net_D.
    fake_Y = net_D(fake_X.detach())  # Discriminator output on generated (fake) samples

    loss_D = (loss(real_Y, real_labels.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()   # Only update the net_D
    return loss_D

def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    # Prepare the labels
    ones = torch.ones((batch_size,), device=Z.device)

    trainer_G.zero_grad()
    fake_X = net_G(Z)  # Generated (fake) samples from noise Z
    # Recomputing `fake_Y` is needed since `net_D` is changed
    # No .detach() here: gradient must flow through net_D back to net_G,
    # so the generator can learn to produce more realistic samples.
    fake_Y = net_D(fake_X)  # Discriminator output on generated (fake) samples

    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()   # Only update the net_D
    return loss_G


class Generator(nn.Module):
    """cGAN generator — transposed convolutions with conditional label channels.

    Input: noise + one-hot labels concatenated along channel dim
           (noise_channels = z_dim + num_classes).
    """

    def __init__(self, noise_channels, img_channels, features):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self.block(noise_channels, features * 64, 4, 1, 0),
            self.block(features * 64, features * 32, 4, 2, 1),
            self.block(features * 32, features * 16, 4, 2, 1),
            self.block(features * 16, features * 8, 4, 2, 1),
            self.block(features * 8, features * 4, 4, 2, 1),
            self.block(features * 4, features * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features * 2, img_channels, kernel_size=4,
                stride=2, padding=1),
            nn.Tanh())

    def block(self, in_channels, out_channels,
              kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """cGAN critic (WGAN-GP) — convolutional discriminator with instance norm.

    Input: images + conditional label channels concatenated along channel dim
           (img_channels = input_img_channels + num_classes).
    """

    def __init__(self, img_channels, features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, features,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.block(features, features * 2, 4, 2, 1),
            self.block(features * 2, features * 4, 4, 2, 1),
            self.block(features * 4, features * 8, 4, 2, 1),
            self.block(features * 8, features * 16, 4, 2, 1),
            self.block(features * 16, features * 32, 4, 2, 1),
            nn.Conv2d(features * 32, 1, kernel_size=4,
                      stride=2, padding=0))

    def block(self, in_channels, out_channels,
              kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.net(x)
