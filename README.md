# llm-utils
Utilities for LLM construction and usage.

LLM Building Blocks.

Upon "Build a Large Language Model (From Scratch)" by Sebastian Raschka".

## LLM building blocks

Utility|Module|Class(es)/Function(s)
---|---|---
Self-attention mechanism classes. | `llm_utils.selfAttn`
Causal-attention mechanism classes. | `llm_utils.causalAttn`|
Activation functions.|`llm_utils.activation`|`GELU`
FeedForward block.|`llm_utils.feedForward`|`FeedForward`
GPT Model.|`llm_utils.GPT`|`GPTModel`
LLM instance.|`llm_utils.LLM`|`LLM`
MultiHeadAttention.|`llm_utils.multiHeadAttention`|`MultiHeadAttention`
Normalization block.|`llm_utils.normalization`|`LayerNorm`
Transformer block.|`ll_utils.transformer`|`TransformerBlock`

## LLM training blocks

Utility|Module|Class/Function
---|---|---
Pytorch DataLoader for LLM model. | `llm_utils.dataLoader`|`createDataLoader`
Pytorch Dataset for LLM model. | `llm_utils.dataSet`|`GPT_Dataset`
Trainer class for LLM model training.|`llm_utils.trainer`|`GPT_Trainer`
Utility colorized print of data batches.|`llm_utils.printDataset`|`print_data_set`

## LLM usage utilities

Utility|Module
---|---

# Documentation

Docs|Remarks
---|---
[Markdown docs](docs-md/docs/index.md)|Generated using [mddocs](https://github.com/mirekfoo/mddocs)
[Web docs](https://mirekfoo.github.io/llm-utils/api/)|Generated using [mkdocs-pyapi](https://github.com/mirekfoo/mkdocs-pyapi)

# Usage

## Install in client project

### pip direct install

```bash
pip install git+https://github.com/mirekfoo/llm-utils.git
```

### pip install upon pyproject.toml

* `pyproject.toml`:
```toml
[project]
dependencies = [    
    "pyutils @ git+https://github.com/mirekfoo/llm-utils.git"
]
```

```bash
pip install .
```

## Install as editable dependency

```bash
git clone https://github.com/mirekfoo/llm-utils.git
pip install -e pyutils
```

# Development

* Type `make help` for available **dev** procedures.
.