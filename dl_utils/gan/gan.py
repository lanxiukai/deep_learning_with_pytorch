"""Shared discriminator and generator update steps used by the basic GAN."""

import torch


def update_D(
    real_samples,
    noise,
    discriminator,
    generator,
    loss,
    optimizer,
    real_label: float = 1.0,
):
    """Update the discriminator on one batch of real and generated samples."""
    batch_size = real_samples.shape[0]
    real_labels = torch.full(
        (batch_size,),
        real_label,
        device=real_samples.device,
    )
    generated_labels = torch.zeros(batch_size, device=real_samples.device)

    optimizer.zero_grad()
    real_outputs = discriminator(real_samples)
    generated_samples = generator(noise)
    generated_outputs = discriminator(generated_samples.detach())
    discriminator_loss = (
        loss(real_outputs, real_labels.reshape(real_outputs.shape))
        + loss(
            generated_outputs,
            generated_labels.reshape(generated_outputs.shape),
        )
    ) / 2
    discriminator_loss.backward()
    optimizer.step()
    return discriminator_loss


def update_G(noise, discriminator, generator, loss, optimizer):
    """Update the generator to make generated samples score as real."""
    batch_size = noise.shape[0]
    real_labels = torch.ones(batch_size, device=noise.device)

    optimizer.zero_grad()
    generated_samples = generator(noise)
    generated_outputs = discriminator(generated_samples)
    generator_loss = loss(
        generated_outputs,
        real_labels.reshape(generated_outputs.shape),
    )
    generator_loss.backward()
    optimizer.step()
    return generator_loss


__all__ = ["update_D", "update_G"]
