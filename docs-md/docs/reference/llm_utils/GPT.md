---
sidebar_label: GPT
title: llm_utils.GPT
---

GPT Model. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 4.

## GPTModel Objects

```python
class GPTModel(nn.Module)
```

Autoregressive GPT-style language model.

This model builds a simple GPT architecture using configurable
transformer blocks. It combines token embeddings with positional embeddings,
applies dropout, passes the result through a stack of transformer layers,
and produces vocabulary logits through a final linear projection.

#### \_\_init\_\_

```python
def __init__(cfg)
```

Initialize the GPT model components.

**Arguments**:

- `cfg` _dict_ - Configuration dictionary containing model hyperparameters.
  Expected keys:
- `"vocab_size"` _int_ - Size of the tokenizer vocabulary.
- `"emb_dim"` _int_ - Embedding dimensionality.
- `"context_length"` _int_ - Maximum input sequence length.
- `"drop_rate"` _float_ - Dropout probability applied to embeddings.
- `"TransformerBlock"` _str, optional_ - Fully qualified class name for
  the transformer block implementation.
- `"n_layers"` _int_ - Number of transformer layers.
- `"LayerNorm"` _str, optional_ - Fully qualified class name for
  normalization layer implementation.

#### forward

```python
def forward(in_idx)
```

Compute output logits for a batch of token sequences.

**Arguments**:

- `in_idx` _torch.LongTensor_ - Input token indices of shape
  [batch_size, seq_len].
  

**Returns**:

- `torch.Tensor` - Logits over the vocabulary with shape
  [batch_size, seq_len, vocab_size].

