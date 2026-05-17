---
sidebar_label: feedForward
title: llm_utils.feedForward
---

FeedForward block. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 4.

## FeedForward Objects

```python
class FeedForward(nn.Module)
```

A feed-forward neural network block used in transformer architectures.

This module consists of two linear layers with an activation function in between,
expanding the embedding dimension by a factor of 4 in the hidden layer.

#### \_\_init\_\_

```python
def __init__(cfg)
```

Initialize the FeedForward module.

**Arguments**:

- `cfg` _dict_ - Configuration dictionary containing:
  - &quot;emb_dim&quot; (int): The embedding dimension.
  - &quot;Activation&quot; (str, optional): The activation function class path.
  Defaults to &quot;llm_utils.activation.GELU&quot;.

#### forward

```python
def forward(x)
```

Perform the forward pass through the feed-forward network.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor of shape (batch_size, seq_len, emb_dim).
  

**Returns**:

- `torch.Tensor` - Output tensor of the same shape as input.

