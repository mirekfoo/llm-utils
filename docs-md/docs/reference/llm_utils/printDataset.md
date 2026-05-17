---
sidebar_label: printDataset
title: llm_utils.printDataset
---

Utility routines for colorized print of data batches.

#### print\_data\_set

```python
def print_data_set(inputs, targets, tokenizer)
```

Prints the input and target data sets in a colorized and formatted manner.

**Arguments**:

- `inputs` _torch.Tensor_ - The input tensor data.
- `targets` _torch.Tensor_ - The target tensor data.
- `tokenizer` _Tokenizer_ - The tokenizer used to decode and colorize the data.
  The function performs the following steps:
  1. Converts the input and target tensors to lists.
  2. Colorizes the input and target data using the tokenizer.
  3. Decodes the input data using the tokenizer.
  4. Calculates the necessary widths for formatting the output.
  5. Prints the input data, colorized input data, and target data in a formatted manner.

#### print\_data\_batch

```python
def print_data_batch(data_batch, tokenizer)
```

Prints a batch of data using the provided tokenizer.

**Arguments**:

- `data_batch` _tuple_ - A tuple containing two elements:
  - inputs: The input data.
  - targets: The target data.
- `tokenizer` - A tokenizer object used to decode the input and target data.
  

**Returns**:

  None

