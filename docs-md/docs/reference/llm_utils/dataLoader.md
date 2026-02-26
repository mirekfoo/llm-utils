---
sidebar_label: dataLoader
title: llm_utils.dataLoader
---

Pytorch DataLoader for LLM model. upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 2.
Utility routines.

## GPTDatasetV1 Objects

```python
class GPTDatasetV1(Dataset)
```

A custom dataset class for preparing text data for GPT model training.

**Arguments**:

- `txt` _str_ - The input text to be tokenized and split into sequences.
- `tokenizer` _Tokenizer_ - The tokenizer to convert text into token IDs.
- `max_length` _int_ - The maximum length of each input sequence.
- `stride` _int_ - The step size to move the window for creating sequences.

**Attributes**:

- `tokenizer` _Tokenizer_ - The tokenizer used for encoding the text.
- `input_ids` _List[torch.Tensor]_ - List of input token ID sequences.
- `target_ids` _List[torch.Tensor]_ - List of target token ID sequences.

**Methods**:

- `__len__()` - Returns the number of sequences in the dataset.
- `__getitem__(idx)` - Returns the input and target sequences at the specified index.

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

