"""
Utilities for multiple OpenAI-compatible LLM providers, including:
- LLMApisManager: manage provider/account API connections.
- ModelSelection: select and cache available models per provider/account.
- Dashboard: build a multi-provider/account model selection dashboard.
- Chat: execute chat completions using the selected provider/model.
"""

import api_keys
from pyutils.json_util import obj2JSON
from pyutils.kwargs import getKwarg

from openai import OpenAI
import json

# --------------------------------------------------------------------------

class LLMApisManager:
    """ Multiple LLMs API providers manager."""

    def __init__(self):
        """Initialize the LLMApisManager with an empty cache of API clients."""
        self.apis = {}

    @staticmethod
    def _connect(provider, account):
        """ Connect to the API of a provider and return the API object.

        Args:
            provider (str): The name of the LLM provider.
            account (str): The account name for the provider.

        Returns:
            OpenAI: The API object for the provider.
        """

        api_key = api_keys.getApiParam(api_keys.API_KEY, provider, account)
        api_url = api_keys.getApiParam(api_keys.API_URL, provider, account)
        return OpenAI(api_key=api_key, base_url=api_url)

    def getApi(self, provider, account):
        """Get or create an API client for the specified provider and account.

        The manager keeps a cache of already connected API clients in
        self.apis. If the requested provider/account pair is not present,
        a new connection is established using the private _connect()
        method and cached for later reuse.

        Args:
            provider (str): The name of the LLM provider.
            account (str): The account name for the provider.

        Returns:
            OpenAI: The connected API client for the provider/account pair.
        """

        if provider not in self.apis:
            self.apis[provider] = {}
        if account not in self.apis[provider]:
            self.apis[provider][account] = self._connect(provider, account)
        return self.apis[provider][account]

    @staticmethod
    def getModelNames(api: OpenAI):
        """Get the list of available model names from the OpenAI API.

        Args:
            api (OpenAI): The OpenAI API client instance.

        Returns:
            list: A list of model IDs (strings).
        """
        return [ model.id for model in api.models.list().data ]


class ModelSelection:
    """ Single LLM model selection."""
    
    def __init__(self, provider, account, apisManager):
        """Initialize the ModelSelection.

        Args:
            provider (str): The name of the LLM provider.
            account (str): The account name for the provider.
            apisManager (LLMApisManager): The API manager instance.
        """
        self.provider = provider
        self.account = account
        self.apisManager = apisManager
        self.models_list = None
        self.selected = None

    @property
    def models(self):
        """Get the list of available models for the provider and account.

        Returns:
            list: List of model names.
        """
        if self.models_list is None:
            self.models_list = self.apisManager.getModelNames(self.apisManager.getApi(self.provider, self.account))
        return self.models_list

# --------------------------------------------------------------------------

class Dashboard:
    """Multiple LLMs model selection dashboard."""
    

    def _build_model_selection(self):
        """Build model selection objects for available providers and accounts.

        Returns:
            dict: Mapping of provider/account keys to ModelSelection instances.
        """

        providers = api_keys.getProviders('LLM')
        providers = [ provider for provider in providers if provider not in self.providers_blacklist ]
        return { provider: ModelSelection(provider, account, self.apisManager) for provider in providers for account in api_keys.getAccounts(provider) } 

    def __init__(self, providers_blacklist=[]):
        """Initialize the dashboard and prepare model selections.

        Args:
            providers_blacklist (list): List of provider names to exclude from the dashboard.
        """

        self.apisManager = LLMApisManager()
        self.providers_blacklist = providers_blacklist
        self.MODELS = self._build_model_selection()
        self.provider = None
        self._model = None
        self.use_history = True

    @property
    def providers(self):
        """Return a list of provider/account keys available in the dashboard."""
        return list(self.MODELS.keys())

    @property
    def model(self):
        """Return the currently selected model name."""
        return self._model

    @model.setter
    def model(self, model):
        """Set the current model and update the selected model for the current provider."""
        self._model = model
        if self.provider is not None:
            self.MODELS[self.provider].selected = model

    @property
    def account(self):
        """Return the account associated with the current provider selection."""
        if not self.provider:
            return None
        if self.provider not in self.MODELS:
            return None
        return self.MODELS[self.provider].account

    def getProviderModels(self, provider):
        """Return the available model names for the specified provider/account."""
        if provider not in self.MODELS:
            return None
        return self.MODELS[provider].models

    def getProviderSelectedModel(self, provider):
        """Return the selected model for the specified provider/account.

        If no model has been selected yet, select and return the first available model.
        """
        if provider not in self.MODELS:
            return None
        if self.MODELS[provider].selected is None:
            self.MODELS[provider].selected = self.MODELS[provider].models[0]
        return self.MODELS[provider].selected


# --------------------------------------------------------------------------

class Chat:
    """Chat loop using currently selected LLM.
    You can switch between model providers during the chat."""

    def __init__(self, dashboard, **kwargs):
        """Initialize the chat interface.

        Args:
            dashboard (Dashboard): Dashboard instance with provider and model selection.
            return_protocol (bool): Whether to include the request/response protocol in results.
            return_history (bool): Whether to return updated conversation history.
        """
        self.dashboard = dashboard
        self.return_protocol = getKwarg(kwargs, 'return_protocol', False)
        self.return_history = getKwarg(kwargs, 'return_history', False)

    def __call__(self, message, history):
        """Send a message to the selected model and return the response.

        Args:
            message (str): The user message to send.
            history (list): Conversation history messages.

        Returns:
            dict or str: Response payload containing answer and optionally history/protocol.
        """

        if (self.dashboard.provider is None or self.dashboard.model is None):
            return "Please select a provider and a model"

        new_messages = [{"role": "user", "content": message}]

        if self.dashboard.use_history:
            query_messages = history + new_messages
        else:
            query_messages = new_messages

        api = self.dashboard.apisManager.getApi(self.dashboard.provider, self.dashboard.account)
        response = api.chat.completions.create(
            model=self.dashboard.model,
            messages=query_messages
        )
        
        answer = response.choices[0].message.content

        res = {}
        res['answer'] = f"[{self.dashboard.provider} / {self.dashboard.model}]\n{answer}"
        
        if self.return_history:
            role = response.choices[0].message.role
            response_record = [{"role": role, "content": answer}]
            new_history = history + new_messages + response_record
            res['history'] = new_history

        if self.return_protocol:
            query_messages_str = json.dumps(query_messages, indent=1, default=obj2JSON)
            response_str = json.dumps(response.model_dump(), indent=2)
            protocol = f"Query: {query_messages_str}\nResponse: {response_str}"
            res['protocol'] = protocol

        return res
