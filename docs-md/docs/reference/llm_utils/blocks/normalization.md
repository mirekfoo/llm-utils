---
sidebar_label: normalization
title: llm_utils.blocks.normalization
---

Normalization block. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 4.

## LayerNorm Objects

```python
class LayerNorm(nn.Module)
```

Layer normalization module.

This module implements layer normalization across the last dimension of the
input tensor. It computes an affine transform of normalized activations
using learnable scale and shift parameters. This is useful in transformer
blocks and other architectures that require input normalization without
depending on batch statistics.

#### \_\_init\_\_

```python
def __init__(emb_dim)
```

Initialize the layer normalization module.

**Arguments**:

- `emb_dim` _int_ - The dimensionality of the last axis of the input
  tensor. This determines the shape of the learnable scale and
  shift parameters.

#### forward

```python
def forward(x)
```

Apply layer normalization to the input tensor.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (..., emb_dim), where the
  last dimension corresponds to the embedding dimension.
  

**Returns**:

- `torch.Tensor` - Normalized tensor with the same shape as the input.

