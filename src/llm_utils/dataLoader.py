"""Pytorch DataLoader for LLM model. upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 2.
    Utility routines."""

import torch
from torch.utils.data import Dataset, DataLoader

import nlp.tokenizer.tiktokenColor as tkncolor

class GPTDatasetV1(Dataset):
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
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def print_data_set(inputs, targets, tokenizer):
    """
    Prints the input and target data sets in a colorized and formatted manner.
    Args:
        inputs (torch.Tensor): The input tensor data.
        targets (torch.Tensor): The target tensor data.
        tokenizer (Tokenizer): The tokenizer used to decode and colorize the data.
    The function performs the following steps:
    1. Converts the input and target tensors to lists.
    2. Colorizes the input and target data using the tokenizer.
    3. Decodes the input data using the tokenizer.
    4. Calculates the necessary widths for formatting the output.
    5. Prints the input data, colorized input data, and target data in a formatted manner.
    """
    input_data_row = inputs.tolist()
    output_data_row = targets.tolist()
    input_data_row_colorized = tkncolor.colorStr(tokenizer, input_data_row).replace('\n', f"{tkncolor.COLOR_RESET_CODE} ")
    output_data_row_colorized = tkncolor.colorStr(tokenizer, output_data_row, 1).replace('\n', f"{tkncolor.COLOR_RESET_CODE} ")

    input_data_row_str = f"{input_data_row}"
    input_data_row_decoded_str = tokenizer.decode(input_data_row)

    tensor_str_len = len(input_data_row_str)
    decoded_str_len = len(input_data_row_decoded_str)

    w1 = max(tensor_str_len, decoded_str_len)
    w2 = 6
    
    input_data_row_ljust_str = input_data_row_str.ljust(w1)
    input_data_row_colorized_str = f"{input_data_row_colorized}"
    input_data_row_colorized_ljust_str = input_data_row_colorized_str + (' ' * (w1-decoded_str_len))
    arrow_str = "->".center(w2)

    print(f"{input_data_row_ljust_str} {arrow_str}{output_data_row}")
    print(f" {input_data_row_colorized_ljust_str}{arrow_str} {output_data_row_colorized}")

def print_data_batch(data_batch, tokenizer):
    """
    Prints a batch of data using the provided tokenizer.

    Args:
        data_batch (tuple): A tuple containing two elements:
            - inputs: The input data.
            - targets: The target data.
        tokenizer: A tokenizer object used to decode the input and target data.

    Returns:
        None
    """
    (inputs, targets) = data_batch
    for i in range(len(inputs)):
        print_data_set(inputs[i], targets[i], tokenizer)