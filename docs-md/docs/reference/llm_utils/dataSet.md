---
sidebar_label: dataSet
title: llm_utils.dataSet
---

Dataset class for LLM model training. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 2.
Utility routines for colorized print of data batches.

## GPT\_Dataset Objects

```python
class GPT_Dataset(Dataset)
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

#### \_\_init\_\_

```python
def __init__(txt, cfg)
```

Initialize dataset: tokenize text and build input/target tensors.

Parameters
----------
txt : str
    Raw text to tokenize.
cfg : mapping-like
    Configuration as described in the class docstring.

#### \_\_len\_\_

```python
def __len__()
```

Return the number of sequence samples in the dataset.

This method implements the Dataset protocol required by PyTorch and
reports how many (input, target) sequence pairs are available. The
length is determined by the number of input_id tensors produced when
the raw text was tokenized and split according to max_length and
stride configuration parameters.

**Arguments**:

  None
  

**Returns**:

- `int` - Number of (input, target) sequence pairs in the dataset.
  

**Notes**:

  The returned value may be zero if the tokenized text was shorter
  than the configured max_length or if no chunks were produced.

#### \_\_getitem\_\_

```python
def __getitem__(idx)
```

Retrieve a single (input, target) tensor pair by index.

**Arguments**:

- `idx` _int_ - Index of the sequence pair to retrieve. Must be in the
  range [0, len(self)).
  

**Returns**:

  tuple[torch.Tensor, torch.Tensor]: A tuple containing the input
  token ids tensor and the corresponding target token ids
  tensor for the given index.
  

**Notes**:

  - Both tensors are created during dataset initialization and are
  returned without further copying.
  - If idx is out of range, the underlying list access will raise
  IndexError.

