""" Utilities for Multiple OpenAI compatible LLMs selection and usage."""

import api_keys
from pyutils.json_util import obj2JSON

from openai import OpenAI
import json

# --------------------------------------------------------------------------

class LLMApisManager:
    """ Multiple LLMs API providers manager."""

    def __init__(self):
        self.apis = {}

    @staticmethod
    def _connect(provider, account):
        api_key = api_keys.getApiParam(api_keys.API_KEY, provider, account)
        api_url = api_keys.getApiParam(api_keys.API_URL, provider, account)
        return OpenAI(api_key=api_key, base_url=api_url)

    def getApi(self, provider, account):
        if provider not in self.apis:
            self.apis[provider] = {}
        if account not in self.apis[provider]:
            self.apis[provider][account] = self._connect(provider, account)
        return self.apis[provider][account]

    @staticmethod
    def getModelNames(api: OpenAI):
        return [ model.id for model in api.models.list().data ]


class ModelSelection:
    """ Single LLM model selection."""
    
    def __init__(self, provider, account, apisManager):
        self.provider = provider
        self.account = account
        self.apisManager = apisManager
        self.models_list = None
        self.selected = None

    @property
    def models(self):
        if self.models_list is None:
            self.models_list = self.apisManager.getModelNames(self.apisManager.getApi(self.provider, self.account))
        return self.models_list

# --------------------------------------------------------------------------

class Dashboard:
    """Multiple LLMs model selection dashboard."""
    

    def _build_model_selection(self):
        providers = api_keys.getProviders('LLM')
        providers = [ provider for provider in providers if provider not in self.providers_blacklist ]
        return { provider: ModelSelection(provider, account, self.apisManager) for provider in providers for account in api_keys.getAccounts(provider) } 

    def __init__(self, providers_blacklist=[]):
        self.apisManager = LLMApisManager()
        self.providers_blacklist = providers_blacklist
        self.MODELS = self._build_model_selection()
        self.provider = None
        self._model = None
        self.use_history = True

    @property
    def providers(self):
        return list(self.MODELS.keys())

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if self.provider is not None:
            self.MODELS[self.provider].selected = model

    @property
    def account(self):
        if not self.provider:
            return None
        if self.provider not in self.MODELS:
            return None
        return self.MODELS[self.provider].account

    def getProviderModels(self, provider):
        if provider not in self.MODELS:
            return None
        return self.MODELS[provider].models

    def getProviderSelectedModel(self, provider):
        if provider not in self.MODELS:
            return None
        if self.MODELS[provider].selected is None:
            self.MODELS[provider].selected = self.MODELS[provider].models[0]
        return self.MODELS[provider].selected


# --------------------------------------------------------------------------

class Chat:
    """Chat loop using currently selected LLM.
    You can switch between model providers during the chat."""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def __call__(self, message, history):

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
        role = response.choices[0].message.role
        
        response_record = [{"role": role, "content": answer}]
        new_history = history + new_messages + response_record

        query_messages_str = json.dumps(query_messages, indent=1, default=obj2JSON)
        response_str = json.dumps(response.model_dump(), indent=2)

        protocol = f"Query: {query_messages_str}\nResponse: {response_str}"

        return f"[{self.dashboard.provider} / {self.dashboard.model}]\n{answer}", protocol, new_history
