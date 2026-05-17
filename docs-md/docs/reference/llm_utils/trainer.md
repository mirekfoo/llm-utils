---
sidebar_label: trainer
title: llm_utils.trainer
---

Trainer class for LLM model training. Upon: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 5.

## GPT\_Trainer Objects

```python
class GPT_Trainer()
```

Trainer class for GPT-style language models.

Handles the complete training pipeline including model optimization, loss computation,
validation, and checkpointing. Implements training loops with support for distributed
learning progress tracking and visualization.

Based on: &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 5.

**Attributes**:

- `llm` - The language model instance to train.
- `cfg` - Configuration dictionary containing training hyperparameters and settings.
- `device` - PyTorch device for computation (CPU or GPU).
- `optimizer` - Optimizer instance for model weight updates.

#### \_\_init\_\_

```python
def __init__(llm, cfg)
```

Initialize the trainer with a language model and configuration.

**Arguments**:

- `llm` - Language model instance with getModel() and getTokenizer() methods.
- `cfg` - Configuration dictionary with training parameters.

#### train\_model

```python
def train_model(text_data)
```

Train the language model on provided text data.

Executes the main training loop over multiple epochs, computing losses,
updating weights, evaluating on validation set, and checkpointing progress.

**Arguments**:

- `text_data` - Text corpus for training.
  

**Returns**:

- `tuple` - (train_losses, val_losses, track_tokens_seen) - lists tracking
  training progress across evaluation steps.

#### plot\_losses

```python
def plot_losses(tokens_seen, train_losses, val_losses, **kwargs)
```

Plot training and validation losses over epochs and tokens seen.

Creates a dual-axis plot showing loss progression against both training epochs
and total tokens processed.

**Arguments**:

- `tokens_seen` _list_ - Tokens processed at each evaluation step.
- `train_losses` _list_ - Training loss values at each evaluation step.
- `val_losses` _list_ - Validation loss values at each evaluation step.
- `**kwargs` - Optional arguments:
- `show` _bool_ - Whether to display the plot. Defaults to False.
- `filename` _str_ - Path to save plot image. Defaults to None (no save).

