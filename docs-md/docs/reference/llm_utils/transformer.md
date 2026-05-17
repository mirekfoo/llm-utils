---
sidebar_label: transformer
title: llm_utils.transformer
---

Transformer block. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 4.

## TransformerBlock Objects

```python
class TransformerBlock(nn.Module)
```

Transformer block combining multi-head attention and feed-forward layers.

This class implements a standard transformer encoder block with layer normalization
applied before each sublayer (pre-normalization) and residual (shortcut) connections
after each sublayer. The architecture follows the design from &quot;Build a Large Language
Model (From Scratch)&quot; by Sebastian Raschka, chapter 4.

**Arguments**:

- `cfg` _dict_ - Configuration dictionary containing:
  - emb_dim (int): Embedding dimension.
  - context_length (int): Maximum context length for attention.
  - n_heads (int): Number of attention heads.
  - drop_rate (float): Dropout rate for regularization.
  - qkv_bias (bool): Whether to use bias in query, key, value projections.

#### \_\_init\_\_

```python
def __init__(cfg)
```

Initialize the transformer block with attention, feed-forward, and normalization layers.

**Arguments**:

- `cfg` _dict_ - Configuration dictionary containing model hyperparameters.

#### forward

```python
def forward(x)
```

Forward pass through the transformer block.

Applies layer normalization, multi-head attention with dropout, then a residual
connection. Subsequently applies layer normalization, feed-forward layer with
dropout, and another residual connection.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape [batch_size, num_tokens, emb_dim].
  

**Returns**:

- `torch.Tensor` - Output tensor of shape [batch_size, num_tokens, emb_dim] after
  applying attention, feed-forward, and residual connections.

