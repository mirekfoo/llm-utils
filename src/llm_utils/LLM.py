"""LLM class. Encapsulates GPTModel upon "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

from pyutils.class_util import (get_class, create_instance)
from pyutils.config_util import read_config_arg
from pyutils.kwargs import getKwarg
import llm_utils.torch.nn.moduleSize as nn_size
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

    def getModelParamsNum(self):
        """Calculates the total number of parameters in the GPT model.

        This method iterates over all parameters in the GPT model and sums
        their sizes to compute the total parameter count, which is a common
        metric for understanding model complexity.

        Returns:
            int: The total number of parameters in the GPT model.
        """
        return nn_size.getModuleParamsNum(self.gpt_model)

    def getModelMemSize(self):
        """Calculates the total memory size of the GPT model parameters in bytes."""
        return nn_size.getModuleMemSize(self.gpt_model)
   
    def getModelBuffersMemSize(self):
        """Calculates the total memory size of the GPT model buffers in bytes."""
        return nn_size.getModuleBuffersMemSize(self.gpt_model)

    def getModel(self):
        """Returns the underlying GPT model instance.

        This method provides access to the GPT model encapsulated within the
        LLM class, allowing for direct interactions if needed.

        Returns:
            The GPT model instance.
        """
        return self.gpt_model
    
    def getTokenizer(self):
        """Returns the tokenizer instance used for encoding and decoding text.

        This method provides access to the tokenizer, which is essential for
        converting between raw text and token IDs that the model can process.

        Returns:
            The tokenizer instance.
        """
        return self.tokenizer
    
    def saveModel(self, path):
        """Saves the GPT model's state dictionary to the specified path.

        This method allows for persisting the trained model weights, enabling
        later loading and inference without retraining.

        Args:
            path (str): The file path where the model state dictionary will be saved.
        """
        torch.save(self.gpt_model.state_dict(), path)

    def loadModel(self, path):
        """Loads the GPT model's state dictionary from the specified path.

        This method allows for restoring a previously saved model, enabling
        continued training or inference.

        Args:
            path (str): The file path from which to load the model state dictionary.
        """
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        self.gpt_model.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache() # usually unnecessary, but can help with memory fragmentation issues on GPU

    def text_encode(self, text: str) -> torch.Tensor:
        """Encodes text into token IDs and returns both tensor and list representations.

        Converts raw text into token indices using the tokenizer, then creates a tensor
        representation with batch dimension. The tensor is moved to the appropriate device
        (CPU or GPU) for model processing.

        Args:
            text (str): The input text to encode.

        Returns:
            tuple: A tuple containing:
                - encoded_tensor (torch.Tensor): Token indices as a 2D tensor with shape (1, seq_len),
                  located on the model's device.
                - encoded (list): Token indices as a list for reference.

        Note:
            The endoftext token '<|endoftext|>' is allowed as a special token during encoding.
        """
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        encoded_tensor = encoded_tensor.to(self._getDevice())
        return (encoded_tensor, encoded)

    def text_decode(self, encoded_tensor : torch.Tensor) -> str:
        """Decodes token IDs back into human-readable text.

        Converts a tensor of token indices into the original text string using the tokenizer.
        Handles batch dimensions by squeezing the tensor before decoding.

        Args:
            encoded_tensor (torch.Tensor): Token indices as a tensor, typically of shape (1, seq_len)
                                          from a single batch example.

        Returns:
            str: The decoded text string.

        Note:
            This method assumes the input tensor is on CPU or will be moved to CPU for decoding.
            The squeeze(0) operation removes the batch dimension before converting to a list.
        """
        return self.tokenizer.decode(encoded_tensor.squeeze(0).tolist())
    
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

        encoded_tensor, encoded = self.text_encode(prompt)

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
            max_new_tokens=self.cfg["max_tokens"],
            context_size=self.cfg["context_length"]
        )
        response = self.text_decode(out)

        if debug_log:
            print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
            print("\nOutput:", out)
            print("Output length:", len(out[0]))
            print("Output text:", response)        

        return response

    def generate_text_simple(self, tokens_batch, max_new_tokens, context_size):
        """Generates text by iteratively predicting the next token.

        Uses greedy decoding to select the token with the highest probability
        at each step. Crops the context to the supported size if necessary.

        Args:
            tokens_batch: Tensor of token indices, shape (batch_size, seq_len).
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
            tokens_batch_context = tokens_batch[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = self.gpt_model(tokens_batch_context)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Get the idx of the vocab entry with the highest logits value
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            tokens_batch = torch.cat((tokens_batch, next_token), dim=1)  # (batch, n_tokens+1)

        return tokens_batch
