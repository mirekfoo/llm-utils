"""Normalization block. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer normalization module.

    This module implements layer normalization across the last dimension of the
    input tensor. It computes an affine transform of normalized activations
    using learnable scale and shift parameters. This is useful in transformer
    blocks and other architectures that require input normalization without
    depending on batch statistics.
    """

    def __init__(self, emb_dim):
        """Initialize the layer normalization module.

        Args:
            emb_dim (int): The dimensionality of the last axis of the input
                tensor. This determines the shape of the learnable scale and
                shift parameters.
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """Apply layer normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim), where the
                last dimension corresponds to the embedding dimension.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input.
        """
        # Compute mean and variance along the embedding dimension.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
