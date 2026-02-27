---
sidebar_label: selfAttn
title: llm_utils.selfAttn
---

Self-attention mechanism classes. upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 3.

## SelfAttention\_Params Objects

```python
class SelfAttention_Params(nn.Module)
```

A simple self-attention mechanism implementation using learnable parameters for query, key, and value projections.

**Arguments**:

- `d_in` _int_ - The input dimension size.
- `d_out` _int_ - The output dimension size for the queries, keys, and values.

#### forward

```python
def forward(x)
```

Forward pass for the SelfAttention_Params module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_len, d_in).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_len, d_out).

## SelfAttention\_Linear Objects

```python
class SelfAttention_Linear(nn.Module)
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

Forward pass for the SelfAttention_Linear module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_len, d_in).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_len, d_out).

## SelfAttention\_Linear4 Objects

```python
class SelfAttention_Linear4(nn.Module)
```

#### forward

```python
def forward(x)
```

Forward pass for the SelfAttention_Linear4 module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_len, d_in).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_len, d_out).

## SelfAttention\_Linear5 Objects

```python
class SelfAttention_Linear5(nn.Module)
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

Forward pass for the SelfAttention_Linear5 module.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_len, d_in).
  

**Returns**:

- `torch.Tensor` - Output tensor of shape (batch_size, seq_len, d_out).

