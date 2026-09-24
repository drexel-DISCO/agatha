"""
Noise injection utilities.

Deterministic Gaussian noise for the embedding-space robustness experiments
of the baseline model (agatha_base.py).
"""

import torch


def add_noise_to_embeddings(embeddings, noise_level, noise_type='absolute', seed=42):
    """
    Add deterministic noise to embeddings for robustness testing.

    Args:
        embeddings: Tensor of shape [batch, seq_len, hidden_size]
        noise_level: Float, amount of noise to add
        noise_type: String, either 'absolute' or 'percentage'
        seed: Int, random seed for deterministic noise generation

    Returns:
        noisy_embeddings: Tensor of same shape as input

    Noise Types:
        - 'absolute': Add Gaussian noise scaled by noise_level
          Formula: embeddings + noise_level * noise

        - 'percentage': Add Gaussian noise scaled by noise_level and embedding std
          Formula: embeddings + (noise_level * embedding_std) * noise
          Uses standard deviation (not L2 norm) for proper statistical scaling
    """
    if noise_level == 0:
        return embeddings

    # Generate deterministic noise using a seeded generator
    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(seed)
    noise = torch.randn(embeddings.shape, generator=generator, device=embeddings.device)

    if noise_type == 'absolute':
        # Absolute noise: direct scaling
        noisy_embeddings = embeddings + noise_level * noise
    elif noise_type == 'percentage':
        # Percentage noise: scale by standard deviation of embeddings
        # Use std() not norm() - std measures variability, which is correct for SNR
        embedding_std = embeddings.std(dim=-1, keepdim=True)
        noisy_embeddings = embeddings + (noise_level * embedding_std) * noise
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Must be 'absolute' or 'percentage'")

    return noisy_embeddings
