"""Pytorch DataLoader for LLM model. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 2.
    Utility routines."""

import torch
from torch.utils.data import Dataset, DataLoader

from pyutils.class_util import (get_class, create_instance)
from pyutils.config_util import read_config_arg

def createDataLoader(txt, cfg):

    context_length = read_config_arg(cfg, "context_length", 1024)
    cfg["max_length"] = context_length
    cfg["stride"] = context_length

    # Create dataset
    GPT_Dataset = get_class(read_config_arg(cfg, "GPT_Dataset", "llm_utils.dataSet.GPT_Dataset"))
    dataset = GPT_Dataset(txt, cfg)

    batch_size = read_config_arg(cfg, "batch_size", 4)
    shuffle = read_config_arg(cfg, "shuffle", True)
    drop_last = read_config_arg(cfg, "drop_last", True)
    num_workers = read_config_arg(cfg, "num_workers", 0)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
