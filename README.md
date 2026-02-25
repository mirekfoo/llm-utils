# llm-utils
Utilities for LLM construction and usage.

## LLM construction utilities

Utility|Module
---|---
Pytorch DataLoader for LLM model. | `llm_utils.dataLoader`
Self-attention mechanism classes. | `llm_utils.selfAttn`
Causal-attention mechanism classes. | `llm_utils.causalAttn`

## LLM usage utilities

Utility|Module
---|---

# Documentation

Docs|Remarks
---|---
[Markdown docs](docs-md/docs/index.md)|Generated using [mddocs](https://github.com/mirekfoo/mddocs)
[Web docs](https://mirekfoo.github.io/pyutils/api/)|Generated using [mkdocs-pyapi](https://github.com/mirekfoo/mkdocs-pyapi)

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