"""GPT Model. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

import torch
import torch.nn as nn
from pyutils.class_util import get_class
from pyutils.config_util import read_config_arg

class GPTModel(nn.Module):
    """Autoregressive GPT-style language model.

    This model builds a simple GPT architecture using configurable
    transformer blocks. It combines token embeddings with positional embeddings,
    applies dropout, passes the result through a stack of transformer layers,
    and produces vocabulary logits through a final linear projection.
    """

    def __init__(self, cfg):
        """Initialize the GPT model components.

        Args:
            cfg (dict): Configuration dictionary containing model hyperparameters.
                Expected keys:
                    "vocab_size" (int): Size of the tokenizer vocabulary.
                    "emb_dim" (int): Embedding dimensionality.
                    "context_length" (int): Maximum input sequence length.
                    "drop_rate" (float): Dropout probability applied to embeddings.
                    "TransformerBlock" (str, optional): Fully qualified class name for
                        the transformer block implementation.
                    "n_layers" (int): Number of transformer layers.
                    "LayerNorm" (str, optional): Fully qualified class name for
                        normalization layer implementation.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        TransformerBlock = get_class(read_config_arg(cfg, "TransformerBlock", "llm_utils.transformer.TransformerBlock"))
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        LayerNorm = get_class(read_config_arg(cfg, "LayerNorm", "llm_utils.normalization.LayerNorm"))
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """Compute output logits for a batch of token sequences.

        Args:
            in_idx (torch.LongTensor): Input token indices of shape
                [batch_size, seq_len].

        Returns:
            torch.Tensor: Logits over the vocabulary with shape
                [batch_size, seq_len, vocab_size].
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
