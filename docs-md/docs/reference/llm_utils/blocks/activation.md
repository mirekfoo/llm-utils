---
sidebar_label: activation
title: llm_utils.blocks.activation
---

Activation function. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 4.

## GELU Objects

```python
class GELU(nn.Module)
```

Approximation of the Gaussian Error Linear Unit (GELU) activation function.

#### \_\_init\_\_

```python
def __init__()
```

Initialize the GELU module.

#### forward

```python
def forward(x)
```

Apply the GELU activation function to the input tensor.

**Arguments**:

- `x` _torch.Tensor_ - Input tensor.
  

**Returns**:

- `torch.Tensor` - Output tensor after applying GELU.

