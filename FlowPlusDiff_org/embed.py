import torch
import torch.nn as nn

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding, self).__init__()

        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)


    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x) 
        
        return x
    

def sinusoidal_embedding(t, dim, max_period=10000):
    """
    t:   (B,)  diffusion steps (Long or Float)
    dim: embedding dimension
    """
    device = t.device
    half = dim // 2

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=device) / half
    )  # (half,)

    args = t.float()[:, None] * freqs[None, :]  # (B, half)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim or dim-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb

class DataEmbeddingcond(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbeddingcond, self).__init__()

        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.d_model=d_model
    def forward(self, x,t):
        x = self.value_embedding(x) + self.position_embedding(x) +sinusoidal_embedding(t, self.d_model).unsqueeze(1)
        return x