"""Activation function. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

import torch
import torch.nn as nn

class GELU(nn.Module):
    """Approximation of the Gaussian Error Linear Unit (GELU) activation function."""

    def __init__(self):
        """Initialize the GELU module."""
        super().__init__()

    def forward(self, x):
        """Apply the GELU activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying GELU.
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
