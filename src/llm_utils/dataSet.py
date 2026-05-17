"""Dataset class for LLM model training. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 2.
    Utility routines for colorized print of data batches."""

import torch
from torch.utils.data import Dataset, DataLoader

from pyutils.class_util import (get_class, create_instance)
from pyutils.config_util import read_config_arg

class GPT_Dataset(Dataset):
    """
    A custom dataset class for preparing text data for GPT model training.
    Args:
        txt (str): The input text to be tokenized and split into sequences.
        tokenizer (Tokenizer): The tokenizer to convert text into token IDs.
        max_length (int): The maximum length of each input sequence.
        stride (int): The step size to move the window for creating sequences.
    Attributes:
        tokenizer (Tokenizer): The tokenizer used for encoding the text.
        input_ids (List[torch.Tensor]): List of input token ID sequences.
        target_ids (List[torch.Tensor]): List of target token ID sequences.
    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Returns the input and target sequences at the specified index.
    """
    def __init__(self, txt, cfg):
        """Initialize dataset: tokenize text and build input/target tensors.

        Parameters
        ----------
        txt : str
            Raw text to tokenize.
        cfg : mapping-like
            Configuration as described in the class docstring.
        """

        self.cfg = cfg
        self.tokenizer = create_instance(read_config_arg(cfg, "Tokenizer", "tiktoken.get_encoding('gpt2')"))

        self.input_ids = []
        self.target_ids = []

        max_length = read_config_arg(cfg, "max_length", 1024)
        stride = read_config_arg(cfg, "stride", 1024)

        token_ids = self.tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Return the number of sequence samples in the dataset.

        This method implements the Dataset protocol required by PyTorch and
        reports how many (input, target) sequence pairs are available. The
        length is determined by the number of input_id tensors produced when
        the raw text was tokenized and split according to max_length and
        stride configuration parameters.

        Args:
            None

        Returns:
            int: Number of (input, target) sequence pairs in the dataset.

        Notes:
            The returned value may be zero if the tokenized text was shorter
            than the configured max_length or if no chunks were produced.
        """
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        """
        Retrieve a single (input, target) tensor pair by index.

        Args:
            idx (int): Index of the sequence pair to retrieve. Must be in the
                range [0, len(self)).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input
                token ids tensor and the corresponding target token ids
                tensor for the given index.

        Notes:
            - Both tensors are created during dataset initialization and are
              returned without further copying.
            - If idx is out of range, the underlying list access will raise
              IndexError.
        """
        # Return the precomputed input and target tensors for the sample.
        return self.input_ids[idx], self.target_ids[idx]
    
