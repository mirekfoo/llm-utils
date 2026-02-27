---
sidebar_label: causalAttn
title: llm_utils.causalAttn
---

Causal-attention mechanism classes. upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 3.5

## CausalAttention\_V0 Objects

```python
class CausalAttention_V0(nn.Module)
```

A simple self-attention mechanism implementation using linear layers for query, key, and value projections.

**Arguments**:

- `d_in` _int_ - The input dimension size.
- `d_out` _int_ - The output dimension size for the queries, keys, and values.
- `qkv_bias` _bool, optional_ - If True, adds a learnable bias to the linear projections. Defaults to False.

#### forward

```python
def forward(x)
```

Forward pass for the CausalAttention_V0 module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_len, d_in).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_len, d_out).

## CausalAttention\_V1 Objects

```python
class CausalAttention_V1(nn.Module)
```

A simple self-attention mechanism implementation using linear layers for query, key, and value projections.

**Arguments**:

- `d_in` _int_ - The input dimension size.
- `d_out` _int_ - The output dimension size for the queries, keys, and values.
- `qkv_bias` _bool, optional_ - If True, adds a learnable bias to the linear projections. Defaults to False.

#### forward

```python
def forward(x)
```

Forward pass for the CausalAttention_V1 module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_len, d_in).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_len, d_out).

