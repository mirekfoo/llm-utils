---
sidebar_label: moduleSize
title: llm_utils.torch.nn.moduleSize
---

torch.nn.Module size calculation utilities.

#### getModuleParamsNum

```python
def getModuleParamsNum(m: torch.nn.Module)
```

Calculates the total number of parameters in the nn.Module.

This method iterates over all parameters in the nn.Module and sums
their sizes to compute the total parameter count, which is a common
metric for understanding model complexity.

**Returns**:

- `int` - The total number of parameters in the nn.Module.

#### getModuleLayerParamNums

```python
def getModuleLayerParamNums(m: torch.nn.Module)
```

Returns a dictionary mapping each parameter name to its number of elements.

#### getModuleMemSize

```python
def getModuleMemSize(m: torch.nn.Module)
```

Calculates the total memory size of the nn.Module parameters in bytes.

#### getModuleLayerMemSizes

```python
def getModuleLayerMemSizes(m: torch.nn.Module)
```

Returns a dictionary mapping each parameter name to its memory size in bytes.

#### getModuleBuffersMemSize

```python
def getModuleBuffersMemSize(m: torch.nn.Module)
```

Calculates the total memory size of the nn.Module buffers in bytes.

#### getModuleLayerBuffersMemSizes

```python
def getModuleLayerBuffersMemSizes(m: torch.nn.Module)
```

Returns a dictionary mapping each buffer name to its memory size in bytes.

