"""FeedForward block. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

import torch.nn as nn
from pyutils.class_util import get_class
from pyutils.config_util import read_config_arg

class FeedForward(nn.Module):
    """A feed-forward neural network block used in transformer architectures.

    This module consists of two linear layers with an activation function in between,
    expanding the embedding dimension by a factor of 4 in the hidden layer.
    """

    def __init__(self, cfg):
        """Initialize the FeedForward module.

        Args:
            cfg (dict): Configuration dictionary containing:
                - "emb_dim" (int): The embedding dimension.
                - "Activation" (str, optional): The activation function class path.
                  Defaults to "llm_utils.blocks.activation.GELU".
        """
        super().__init__()

        Activation = get_class(read_config_arg(cfg, "Activation", "llm_utils.blocks.activation.GELU"))
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            Activation(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """Perform the forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        return self.layers(x)