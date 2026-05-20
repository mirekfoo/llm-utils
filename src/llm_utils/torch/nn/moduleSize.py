"""torch.nn.Module size calculation utilities."""

import torch

def getModuleParamsNum(m : torch.nn.Module):
    """Calculates the total number of parameters in the nn.Module.

    This method iterates over all parameters in the nn.Module and sums
    their sizes to compute the total parameter count, which is a common
    metric for understanding model complexity.

    Returns:
        int: The total number of parameters in the nn.Module.
    """
    return sum(p.numel() for p in m.parameters())

def getModuleLayerParamNums(m : torch.nn.Module):
    """Returns a dictionary mapping each parameter name to its number of elements."""
    return {name: p.numel() for name, p in m.named_parameters()}

def getModuleMemSize(m : torch.nn.Module):
    """Calculates the total memory size of the nn.Module parameters in bytes."""
    return sum(p.numel() * p.element_size() for p in m.parameters())

def getModuleLayerMemSizes(m : torch.nn.Module):
    """Returns a dictionary mapping each parameter name to its memory size in bytes."""
    return {name: p.numel() * p.element_size() for name, p in m.named_parameters()}

def getModuleBuffersMemSize(m : torch.nn.Module):
    """Calculates the total memory size of the nn.Module buffers in bytes."""
    return sum(b.numel() * b.element_size() for b in m.buffers())

def getModuleLayerBuffersMemSizes(m : torch.nn.Module):
    """Returns a dictionary mapping each buffer name to its memory size in bytes."""
    return {name: b.numel() * b.element_size() for name, b in m.named_buffers()}
