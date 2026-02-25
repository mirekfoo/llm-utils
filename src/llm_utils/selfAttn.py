"""Self-attention mechanism classes. upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 3."""

import torch
import torch.nn as nn

class SelfAttention_Params(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

class SelfAttention_Linear(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        self.values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        self.attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        self.context_vec = self.attn_weights @ self.values
        return self.context_vec
    
class SelfAttention_Linear4(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        self.values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        self.attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # context_vec = self.attn_weights @ self.values
        self.attnValues2ContextL = nn.Linear(self.attn_weights.shape[0], self.values.shape[0], bias=False)
        self.attnValues2ContextL.weight = nn.Parameter(self.attn_weights,requires_grad=False)
        context_vec = self.attnValues2ContextL(self.values.T)

        self.model = self.attnValues2ContextL
        return context_vec.T

class SelfAttention_Linear5(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        self.values = self.W_value(x)
        
        #attn_scores = queries @ keys.T
        self.attnQueries2ScoresL = nn.Linear(queries.shape[1], queries.shape[0], bias=False)
        self.attnQueries2ScoresL.weight = nn.Parameter(keys, requires_grad=False)
        attn_scores = self.attnQueries2ScoresL(queries)
        self.attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # context_vec = self.attn_weights @ self.values
        self.attnValues2ContextL = nn.Linear(self.attn_weights.shape[0], self.values.shape[0], bias=False)
        self.attnValues2ContextL.weight = nn.Parameter(self.attn_weights, requires_grad=False)
        context_vec = self.attnValues2ContextL(self.values.T)

        self.model = self.attnValues2ContextL
        return context_vec.T
