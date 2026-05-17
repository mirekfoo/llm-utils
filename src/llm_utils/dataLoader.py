"""Pytorch DataLoader for LLM model. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 2.
    Utility routines."""

import torch
from torch.utils.data import Dataset, DataLoader

from pyutils.class_util import (get_class, create_instance)
from pyutils.config_util import read_config_arg

def createDataLoader(txt, cfg):
    """Create a PyTorch DataLoader for LLM model training.
    
    Initializes a DataLoader with a GPT dataset using configuration parameters.
    The function dynamically instantiates a dataset class and wraps it in a DataLoader
    with specified batch processing and shuffling options.
    
    Args:
        txt (str): Input text data for the dataset.
        cfg (dict): Configuration dictionary containing parameters for dataset and dataloader.
            Expected keys:
            - context_length (int, optional): Maximum sequence length. Defaults to 1024.
            - GPT_Dataset (str, optional): Class path for dataset. Defaults to 
              "llm_utils.dataSet.GPT_Dataset".
            - batch_size (int, optional): Number of samples per batch. Defaults to 4.
            - shuffle (bool, optional): Whether to shuffle data. Defaults to True.
            - drop_last (bool, optional): Drop incomplete batches. Defaults to True.
            - num_workers (int, optional): Number of data loading workers. Defaults to 0.
    
    Returns:
        torch.utils.data.DataLoader: Configured DataLoader for model training.
    """
    # Extract context length and set max_length and stride for dataset configuration
    context_length = read_config_arg(cfg, "context_length", 1024)
    cfg["max_length"] = context_length
    cfg["stride"] = context_length

    # Dynamically load and instantiate the GPT dataset class
    GPT_Dataset = get_class(read_config_arg(cfg, "GPT_Dataset", "llm_utils.dataSet.GPT_Dataset"))
    dataset = GPT_Dataset(txt, cfg)

    # Extract dataloader configuration parameters
    batch_size = read_config_arg(cfg, "batch_size", 4)
    shuffle = read_config_arg(cfg, "shuffle", True)
    drop_last = read_config_arg(cfg, "drop_last", True)
    num_workers = read_config_arg(cfg, "num_workers", 0)

    # Create and return the configured dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
