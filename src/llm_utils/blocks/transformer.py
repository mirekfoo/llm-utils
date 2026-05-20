"""Transformer block. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

import torch.nn as nn
from pyutils.class_util import get_class
from pyutils.config_util import read_config_arg

class TransformerBlock(nn.Module):
    """Transformer block combining multi-head attention and feed-forward layers.
    
    This class implements a standard transformer encoder block with layer normalization
    applied before each sublayer (pre-normalization) and residual (shortcut) connections
    after each sublayer. The architecture follows the design from "Build a Large Language
    Model (From Scratch)" by Sebastian Raschka, chapter 4.
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - emb_dim (int): Embedding dimension.
            - context_length (int): Maximum context length for attention.
            - n_heads (int): Number of attention heads.
            - drop_rate (float): Dropout rate for regularization.
            - qkv_bias (bool): Whether to use bias in query, key, value projections.
    """
    
    def __init__(self, cfg):
        """Initialize the transformer block with attention, feed-forward, and normalization layers.
        
        Args:
            cfg (dict): Configuration dictionary containing model hyperparameters.
        """
        super().__init__()

        # Initialize multi-head attention layer
        MultiHeadAttention = get_class(read_config_arg(cfg, "Attention", "llm_utils.blocks.attention.MHA.MultiHeadAttention"))
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])

        # Initialize feed-forward network
        FeedForward = get_class(read_config_arg(cfg, "FeedForward", "llm_utils.blocks.feedForward.FeedForward"))    
        self.ff = FeedForward(cfg)

        # Initialize layer normalization layers
        LayerNorm = get_class(read_config_arg(cfg, "LayerNorm", "llm_utils.blocks.normalization.LayerNorm"))
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout layer for regularization of residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """Forward pass through the transformer block.
        
        Applies layer normalization, multi-head attention with dropout, then a residual
        connection. Subsequently applies layer normalization, feed-forward layer with
        dropout, and another residual connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_tokens, emb_dim].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_tokens, emb_dim] after
                applying attention, feed-forward, and residual connections.
        """
        # First sublayer: Multi-head attention with pre-normalization and residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        # Second sublayer: Feed-forward network with pre-normalization and residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        return x