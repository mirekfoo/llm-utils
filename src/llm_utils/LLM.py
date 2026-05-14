"""LLM class. Encapsulates GPTModel upon "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 4."""

from pyutils.class_util import (get_class, create_instance)
from pyutils.config_util import read_config_arg
from pyutils.kwargs import getKwarg
import torch

class LLM:
    def __init__(self, cfg):

        self.cfg = cfg

        self.tokenizer = create_instance(read_config_arg(cfg, "Tokenizer", "tiktoken.get_encoding('gpt2')"))
        #tokenizer = tiktoken.get_encoding("gpt2")

        GPTModel = get_class(read_config_arg(cfg, "GPTModel", "llm_utils.GPT.GPTModel"))
        self.gpt_model = GPTModel(cfg)

    def query(self, prompt: str, **kwargs) -> str:

        debug_log = getKwarg(kwargs, 'debug_log')

        self.gpt_model.eval()  # disable dropout

        encoded = self.tokenizer.encode(prompt)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)

        if debug_log:
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
