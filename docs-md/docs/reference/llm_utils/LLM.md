---
sidebar_label: LLM
title: llm_utils.LLM
---

LLM class. Encapsulates GPTModel upon &quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 4.

## LLM Objects

```python
class LLM()
```

LLM class for interacting with a GPT-based language model.

This class encapsulates a GPT model, providing methods for tokenization,
text generation, and querying the model with prompts. It is based on
&quot;Build a Large Language Model (From Scratch)&quot; by Sebastian Raschka, chapter 4.

**Attributes**:

- `cfg` - Configuration dictionary containing model parameters.
- `tokenizer` - Tokenizer instance for encoding and decoding text.
- `gpt_model` - Instance of the GPT model.

#### \_\_init\_\_

```python
def __init__(cfg)
```

Initializes the LLM instance.

**Arguments**:

- `cfg` - Configuration dictionary containing model settings,
  such as tokenizer and model class paths.

#### getModelParamsNum

```python
def getModelParamsNum()
```

Calculates the total number of parameters in the GPT model.

This method iterates over all parameters in the GPT model and sums
their sizes to compute the total parameter count, which is a common
metric for understanding model complexity.

**Returns**:

- `int` - The total number of parameters in the GPT model.

#### getModelLayerParamNums

```python
def getModelLayerParamNums()
```

Returns a dictionary mapping each parameter name to its number of elements.

#### getModelMemSize

```python
def getModelMemSize()
```

Calculates the total memory size of the GPT model parameters in bytes.

#### getModelLayerMemSizes

```python
def getModelLayerMemSizes()
```

Returns a dictionary mapping each parameter name to its memory size in bytes.

#### getModelBuffersMemSize

```python
def getModelBuffersMemSize()
```

Calculates the total memory size of the GPT model buffers in bytes.

#### getModel

```python
def getModel()
```

Returns the underlying GPT model instance.

This method provides access to the GPT model encapsulated within the
LLM class, allowing for direct interactions if needed.

**Returns**:

  The GPT model instance.

#### getTokenizer

```python
def getTokenizer()
```

Returns the tokenizer instance used for encoding and decoding text.

This method provides access to the tokenizer, which is essential for
converting between raw text and token IDs that the model can process.

**Returns**:

  The tokenizer instance.

#### saveModel

```python
def saveModel(path)
```

Saves the GPT model&#x27;s state dictionary to the specified path.

This method allows for persisting the trained model weights, enabling
later loading and inference without retraining.

**Arguments**:

- `path` _str_ - The file path where the model state dictionary will be saved.

#### loadModel

```python
def loadModel(path)
```

Loads the GPT model&#x27;s state dictionary from the specified path.

This method allows for restoring a previously saved model, enabling
continued training or inference.

**Arguments**:

- `path` _str_ - The file path from which to load the model state dictionary.

#### text\_encode

```python
def text_encode(text: str) -> torch.Tensor
```

Encodes text into token IDs and returns both tensor and list representations.

Converts raw text into token indices using the tokenizer, then creates a tensor
representation with batch dimension. The tensor is moved to the appropriate device
(CPU or GPU) for model processing.

**Arguments**:

- `text` _str_ - The input text to encode.
  

**Returns**:

- `tuple` - A tuple containing:
  - encoded_tensor (torch.Tensor): Token indices as a 2D tensor with shape (1, seq_len),
  located on the model&#x27;s device.
  - encoded (list): Token indices as a list for reference.
  

**Notes**:

  The endoftext token &#x27;&lt;|endoftext|&gt;&#x27; is allowed as a special token during encoding.

#### text\_decode

```python
def text_decode(encoded_tensor: torch.Tensor) -> str
```

Decodes token IDs back into human-readable text.

Converts a tensor of token indices into the original text string using the tokenizer.
Handles batch dimensions by squeezing the tensor before decoding.

**Arguments**:

- `encoded_tensor` _torch.Tensor_ - Token indices as a tensor, typically of shape (1, seq_len)
  from a single batch example.
  

**Returns**:

- `str` - The decoded text string.
  

**Notes**:

  This method assumes the input tensor is on CPU or will be moved to CPU for decoding.
  The squeeze(0) operation removes the batch dimension before converting to a list.

#### query

```python
def query(prompt: str, **kwargs) -> str
```

Queries the model with a prompt and generates a response.

Encodes the prompt, generates new tokens using the model, and decodes
the output back to text. Supports debug logging for inspection.

**Arguments**:

- `prompt` _str_ - The input text prompt to generate a response for.
- `**kwargs` - Additional keyword arguments. Supports &#x27;debug_log&#x27; (bool)
  to enable debug output.
  

**Returns**:

- `str` - The generated response text, including the original prompt.

#### generate\_text\_simple

```python
def generate_text_simple(tokens_batch, max_new_tokens, context_size)
```

Generates text by iteratively predicting the next token.

Uses greedy decoding to select the token with the highest probability
at each step. Crops the context to the supported size if necessary.

**Arguments**:

- `tokens_batch` - Tensor of token indices, shape (batch_size, seq_len).
- `max_new_tokens` _int_ - Maximum number of new tokens to generate.
- `context_size` _int_ - Maximum context length supported by the model.
  

**Returns**:

- `Tensor` - The updated token indices tensor, shape (batch_size, seq_len + max_new_tokens).

