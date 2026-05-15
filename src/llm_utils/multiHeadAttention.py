"""MultiHeadAttention. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 3."""

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for transformer architectures.
    
    This class implements scaled dot-product attention with multiple heads,
    allowing the model to attend to information from different representation
    subspaces at different positions. Includes causal masking to prevent 
    attending to future tokens.
    
    Attributes:
        d_out (int): Output dimension (must be divisible by num_heads).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head (d_out // num_heads).
        W_query (nn.Linear): Linear projection for queries.
        W_key (nn.Linear): Linear projection for keys.
        W_value (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear projection to combine head outputs.
        dropout (nn.Dropout): Dropout layer applied to attention weights.
        mask (torch.Tensor): Causal mask buffer to prevent attending to future tokens.
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """Initialize the multi-head attention module.
        
        Args:
            d_in (int): Input dimension.
            d_out (int): Output dimension. Must be divisible by num_heads.
            context_length (int): Maximum sequence length for causal masking.
            dropout (float): Dropout probability for attention weights.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): Whether to use bias in query, key, and value 
                projections. Defaults to False.
                
        Raises:
            AssertionError: If d_out is not divisible by num_heads.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """Compute multi-head self-attention with causal masking.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).
                
        Returns:
            torch.Tensor: Attention output of shape (batch_size, num_tokens, d_out).
                Represents the context-aware representation of each token.
        """
        b, num_tokens, d_in = x.shape

        # Project input to query, key, and value representations
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # This allows us to compute attention independently for each head
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Scaled dot-product: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # This creates a lower triangular mask to prevent attending to future tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores with -inf for future positions
        # softmax will convert -inf to 0 probability
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Apply softmax and scaling factor (1 / sqrt(head_dim))
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec