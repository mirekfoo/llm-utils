---
sidebar_label: OpenAI
title: llm_utils.multi.OpenAI
---

Utilities for multiple OpenAI-compatible LLM providers. Including:
- LLMApisManager: manage provider/account API connections.
- ModelSelection: select and cache available models per provider/account.
- Dashboard: build a multi-provider/account model selection dashboard.
- Chat: execute chat completions using the selected provider/model.

## LLMApisManager Objects

```python
class LLMApisManager()
```

Multiple LLMs API providers manager.

#### \_\_init\_\_

```python
def __init__()
```

Initialize the LLMApisManager with an empty cache of API clients.

#### getApi

```python
def getApi(provider, account)
```

Get or create an API client for the specified provider and account.

The manager keeps a cache of already connected API clients in
self.apis. If the requested provider/account pair is not present,
a new connection is established using the private _connect()
method and cached for later reuse.

**Arguments**:

- `provider` _str_ - The name of the LLM provider.
- `account` _str_ - The account name for the provider.
  

**Returns**:

- `OpenAI` - The connected API client for the provider/account pair.

#### getModelNames

```python
@staticmethod
def getModelNames(api: OpenAI)
```

Get the list of available model names from the OpenAI API.

**Arguments**:

- `api` _OpenAI_ - The OpenAI API client instance.
  

**Returns**:

- `list` - A list of model IDs (strings).

## ModelSelection Objects

```python
class ModelSelection()
```

Single LLM model selection.

#### \_\_init\_\_

```python
def __init__(provider, account, apisManager)
```

Initialize the ModelSelection.

**Arguments**:

- `provider` _str_ - The name of the LLM provider.
- `account` _str_ - The account name for the provider.
- `apisManager` _LLMApisManager_ - The API manager instance.

#### models

```python
@property
def models()
```

Get the list of available models for the provider and account.

**Returns**:

- `list` - List of model names.

## Dashboard Objects

```python
class Dashboard()
```

Multiple LLMs model selection dashboard.

#### \_\_init\_\_

```python
def __init__(providers_blacklist=[])
```

Initialize the dashboard and prepare model selections.

**Arguments**:

- `providers_blacklist` _list_ - List of provider names to exclude from the dashboard.

#### providers

```python
@property
def providers()
```

Return a list of provider/account keys available in the dashboard.

#### model

```python
@property
def model()
```

Return the currently selected model name.

#### model

```python
@model.setter
def model(model)
```

Set the current model and update the selected model for the current provider.

#### account

```python
@property
def account()
```

Return the account associated with the current provider selection.

#### getProviderModels

```python
def getProviderModels(provider)
```

Return the available model names for the specified provider/account.

#### getProviderSelectedModel

```python
def getProviderSelectedModel(provider)
```

Return the selected model for the specified provider/account.

If no model has been selected yet, select and return the first available model.

## Chat Objects

```python
class Chat()
```

Chat loop using currently selected LLM.
You can switch between model providers during the chat.

#### \_\_init\_\_

```python
def __init__(dashboard, **kwargs)
```

Initialize the chat interface.

**Arguments**:

- `dashboard` _Dashboard_ - Dashboard instance with provider and model selection.
- `return_protocol` _bool_ - Whether to include the request/response protocol in results.
- `return_history` _bool_ - Whether to return updated conversation history.

#### \_\_call\_\_

```python
def __call__(message, history)
```

Send a message to the selected model and return the response.

**Arguments**:

- `message` _str_ - The user message to send.
- `history` _list_ - Conversation history messages.
  

**Returns**:

  dict or str: Response payload containing answer and optionally history/protocol.

