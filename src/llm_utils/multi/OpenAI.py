""" Utilities for Multiple OpenAI compatible LLMs selection and usage."""

import api_keys
from openai import OpenAI

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

apisManager = LLMApisManager()

class ModelSelection:
    """ Single LLM model selection."""
    
    def __init__(self, provider, account):
        self.provider = provider
        self.account = account
        self.models_list = None
        self.selected = None

    @property
    def models(self):
        if self.models_list is None:
            self.models_list = apisManager.getModelNames(apisManager.getApi(self.provider, self.account))
        return self.models_list

# --------------------------------------------------------------------------

class Dashboard:
    """Multiple LLMs model selection dashboard."""
    
    #providers_blacklist = ['OpenAI', 'Anthropic']

    @staticmethod
    def _build_model_selection():
        providers = api_keys.getProviders('LLM')
        providers = [ provider for provider in providers if provider not in Dashboard.providers_blacklist ]
        return { provider: ModelSelection(provider, account) for provider in providers for account in api_keys.getAccounts(provider) } 

    def __init__(self, providers_blacklist=None):
        self.providers_blacklist = providers_blacklist
        self.MODELS = Dashboard._build_model_selection()
        self.provider = None
        self._model = None

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

dashboard = Dashboard()

# --------------------------------------------------------------------------

class Chat:
    """Chat loop using currently selected LLM.
    You can switch between model providers during the chat."""

    def __call__(message, history):
        #print(here())
        #return f"[{dashboard.provider} / {dashboard.model}] → {message}"

        if (dashboard.provider is None or dashboard.model is None):
            return "Please select a provider and a model"

        messages = [{"role": "user", "content": message}]

        api = apisManager.getApi(dashboard.provider, dashboard.account)
        response = api.chat.completions.create(
            model=dashboard.model,
            messages=messages
        )
        answer = response.choices[0].message.content

        return f"[{dashboard.provider} / {dashboard.model}]\n{answer}"
