---
sidebar_label: dataLoader
title: llm_utils.dataLoader
---

Pytorch DataLoader for LLM model. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 2.
Utility routines.

#### createDataLoader

```python
def createDataLoader(txt, cfg)
```

Create a PyTorch DataLoader for LLM model training.

Initializes a DataLoader with a GPT dataset using configuration parameters.
The function dynamically instantiates a dataset class and wraps it in a DataLoader
with specified batch processing and shuffling options.

**Arguments**:

- `txt` _str_ - Input text data for the dataset.
- `cfg` _dict_ - Configuration dictionary containing parameters for dataset and dataloader.
  Expected keys:
  - context_length (int, optional): Maximum sequence length. Defaults to 1024.
  - GPT_Dataset (str, optional): Class path for dataset. Defaults to
  &quot;llm_utils.dataSet.GPT_Dataset&quot;.
  - batch_size (int, optional): Number of samples per batch. Defaults to 4.
  - shuffle (bool, optional): Whether to shuffle data. Defaults to True.
  - drop_last (bool, optional): Drop incomplete batches. Defaults to True.
  - num_workers (int, optional): Number of data loading workers. Defaults to 0.
  

**Returns**:

- `torch.utils.data.DataLoader` - Configured DataLoader for model training.

