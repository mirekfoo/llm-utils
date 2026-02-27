"""Causal-attention mechanism classes. upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 3.5"""

import torch
import torch.nn as nn

class CausalAttention_V0(nn.Module): # = SelfAttention_Linear
    """
    A simple self-attention mechanism implementation using linear layers for query, key, and value projections.

    Args:
        d_in (int): The input dimension size.
        d_out (int): The output dimension size for the queries, keys, and values.
        qkv_bias (bool, optional): If True, adds a learnable bias to the linear projections. Defaults to False.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        Forward pass for the CausalAttention_V0 module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out).
        """
        self.keys = self.W_key(x)
        self.queries = self.W_query(x)
        self.values = self.W_value(x)
        
        self.attn_scores = self.queries @ self.keys.T
        self.attn_weights = torch.softmax(self.attn_scores / self.keys.shape[-1]**0.5, dim=-1)

        self.context_vec = self.attn_weights @ self.values
        return self.context_vec
    

class CausalAttention_V1(nn.Module): 
    """
    A simple self-attention mechanism implementation using linear layers for query, key, and value projections.

    Args:
        d_in (int): The input dimension size.
        d_out (int): The output dimension size for the queries, keys, and values.
        qkv_bias (bool, optional): If True, adds a learnable bias to the linear projections. Defaults to False.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        Forward pass for the CausalAttention_V1 module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out).
        """
        self.keys = self.W_key(x)
        self.queries = self.W_query(x)
        self.values = self.W_value(x)
        
        self.attn_scores = self.queries @ self.keys.T

        context_length = self.attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.attn_scores_masked = self.attn_scores.masked_fill(mask.bool(), -torch.inf)

        self.attn_weights = torch.softmax(self.attn_scores_masked / self.keys.shape[-1]**0.5, dim=1)

        self.context_vec = self.attn_weights @ self.values
        return self.context_vec 