"""LLM class. Encapsulates GPTModel upon "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

from pyutils.class_util import (get_class, create_instance)
from pyutils.config_util import read_config_arg
from pyutils.kwargs import getKwarg
import torch

class LLM:
    """LLM class for interacting with a GPT-based language model.

    This class encapsulates a GPT model, providing methods for tokenization,
    text generation, and querying the model with prompts. It is based on
    "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4.

    Attributes:
        cfg: Configuration dictionary containing model parameters.
        tokenizer: Tokenizer instance for encoding and decoding text.
        gpt_model: Instance of the GPT model.
    """

    def __init__(self, cfg):
        """Initializes the LLM instance.

        Args:
            cfg: Configuration dictionary containing model settings,
                 such as tokenizer and model class paths.
        """
        self.cfg = cfg

        self.tokenizer = create_instance(read_config_arg(cfg, "Tokenizer", "tiktoken.get_encoding('gpt2')"))
        #tokenizer = tiktoken.get_encoding("gpt2")

        GPTModel = get_class(read_config_arg(cfg, "GPTModel", "llm_utils.GPT.GPTModel"))
        self.gpt_model = GPTModel(cfg)
        
        device = self._getDevice()
        self.gpt_model.to(device)

    def _getDevice(self):
        """Determines the device to run the model on based on configuration.

        Checks the 'device' key in the configuration dictionary, defaulting to
        'cpu' if not specified. This allows for flexible deployment on either
        CPU or GPU.

        Returns:
            torch.device: The device to use for model computations.
        """
        return torch.device(read_config_arg(self.cfg, "device", "cpu"))
    
    def query(self, prompt: str, **kwargs) -> str:
        """Queries the model with a prompt and generates a response.

        Encodes the prompt, generates new tokens using the model, and decodes
        the output back to text. Supports debug logging for inspection.

        Args:
            prompt (str): The input text prompt to generate a response for.
            **kwargs: Additional keyword arguments. Supports 'debug_log' (bool)
                      to enable debug output.

        Returns:
            str: The generated response text, including the original prompt.
        """
        debug_log = getKwarg(kwargs, 'debug_log')

        self.gpt_model.eval()  # disable dropout

        encoded = self.tokenizer.encode(prompt)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        encoded_tensor = encoded_tensor.to(self._getDevice())

        if debug_log:
            print(f"\n{50*'='}\n{22*' '}CONFIG\n{50*'='}")
            print(f"Device: {self._getDevice()}")

            print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
            print("\nInput text:", prompt)
            print("Encoded input text:", encoded)
            print("encoded_tensor.shape:", encoded_tensor.shape)

        out = self._generate_text_simple(
            #model=model,
            idx=encoded_tensor,
            max_new_tokens=10,
            context_size=self.cfg["context_length"]
        )
        response = self.tokenizer.decode(out.squeeze(0).tolist())

        if debug_log:
            print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
            print("\nOutput:", out)
            print("Output length:", len(out[0]))
            print("Output text:", response)        

        return response

    def _generate_text_simple(self, idx, max_new_tokens, context_size):
        """Generates text by iteratively predicting the next token.

        Uses greedy decoding to select the token with the highest probability
        at each step. Crops the context to the supported size if necessary.

        Args:
            idx: Tensor of token indices, shape (batch_size, seq_len).
            max_new_tokens (int): Maximum number of new tokens to generate.
            context_size (int): Maximum context length supported by the model.

        Returns:
            Tensor: The updated token indices tensor, shape (batch_size, seq_len + max_new_tokens).
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = self.gpt_model(idx_cond)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx
