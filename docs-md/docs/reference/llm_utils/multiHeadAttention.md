---
sidebar_label: multiHeadAttention
title: llm_utils.multiHeadAttention
---

MultiHeadAttention. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 3.

## MultiHeadAttention Objects

```python
class MultiHeadAttention(nn.Module)
```

Multi-head self-attention mechanism for transformer architectures.

This class implements scaled dot-product attention with multiple heads,
allowing the model to attend to information from different representation
subspaces at different positions. Includes causal masking to prevent
attending to future tokens.

**Attributes**:

- `d_out` _int_ - Output dimension (must be divisible by num_heads).
- `num_heads` _int_ - Number of attention heads.
- `head_dim` _int_ - Dimension of each attention head (d_out // num_heads).
- `W_query` _nn.Linear_ - Linear projection for queries.
- `W_key` _nn.Linear_ - Linear projection for keys.
- `W_value` _nn.Linear_ - Linear projection for values.
- `out_proj` _nn.Linear_ - Linear projection to combine head outputs.
- `dropout` _nn.Dropout_ - Dropout layer applied to attention weights.
- `mask` _torch.Tensor_ - Causal mask buffer to prevent attending to future tokens.

#### \_\_init\_\_

```python
def __init__(d_in, d_out, context_length, dropout, num_heads, qkv_bias=False)
```

Initialize the multi-head attention module.

**Arguments**:

- `d_in` _int_ - Input dimension.
- `d_out` _int_ - Output dimension. Must be divisible by num_heads.
- `context_length` _int_ - Maximum sequence length for causal masking.
- `dropout` _float_ - Dropout probability for attention weights.
- `num_heads` _int_ - Number of attention heads.
- `qkv_bias` _bool, optional_ - Whether to use bias in query, key, and value
  projections. Defaults to False.
  

**Raises**:

- `AssertionError` - If d_out is not divisible by num_heads.

#### forward

```python
def forward(x)
```

Compute multi-head self-attention with causal masking.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, num_tokens, d_in).
  

**Returns**:

- `torch.Tensor` - Attention output of shape (batch_size, num_tokens, d_out).
  Represents the context-aware representation of each token.

