"""Transformer block. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

import torch.nn as nn
from pyutils.class_util import get_class
from pyutils.config_util import read_config_arg

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        MultiHeadAttention = get_class(read_config_arg(cfg, "Attention", "llm_utils.multiHeadAttention.MultiHeadAttention"))
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])

        FeedForward = get_class(read_config_arg(cfg, "FeedForward", "llm_utils.feedForward.FeedForward"))    
        self.ff = FeedForward(cfg)

        LayerNorm = get_class(read_config_arg(cfg, "LayerNorm", "llm_utils.normalization.LayerNorm"))
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x