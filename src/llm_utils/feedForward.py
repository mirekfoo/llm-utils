"""FeedForward block. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

import torch.nn as nn
from pyutils.class_util import get_class
from pyutils.config_util import read_config_arg

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        Activation = get_class(read_config_arg(cfg, "Activation", "llm_utils.activation.GELU"))
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            Activation(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)