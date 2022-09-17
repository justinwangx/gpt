import torch
import torch.nn as nn
import math

# class PositionalEmbedding(nn.Module):
#     """GPT uses learned positional embeddings"""
#     def __init__(self, cfg):
#         super().__init__()

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        assert cfg.dm % cfg.n_heads == 0, 'dimension of model not divisible by number of heads'
        # i'm setting dv = dk = dm / nh, as this is what's typically done
        # this makes the computational cost of MHA similiar to that of attention for a single head
        # but if we wanted to we could set dk and dv to different values
        super().__init__()
        self.dm = cfg.dm
        self.dk = cfg.dm // cfg.n_heads
        self.qkv_proj = nn.Linear(self.dm, self.dm * 3)
        self.output = nn.Linear(self.dm, self.dm)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)
        self.mask = torch.tril(torch.ones(1, 1, cfg.n_positions, cfg.n_positions))

    def forward(self, x):
        bs, seq_len, dm = x.shape
        q, k, v = self.qkv_proj(x).split(self.dm, dim=-1) # (bs, seq_len, dk)
        q, k, v = [x.view(bs, seq_len, -1, self.dk).transpose(1, 2) for x in (q, k, v)] # (bs, nh, seq_len, dk)

        w = q @ k.transpose(-2, -1) / math.sqrt(self.dk) # (bs, nh, seq_len, seq_len)
        w.masked_fill_(self.mask[:, :, :seq_len, :seq_len] == 0, -float('inf'))
        w = self.softmax(w)
        w = self.attn_dropout(w)

        z = w @ v # (bs, nh, seq_len, dk)
        z = z.transpose(1, 2).contiguous().view(bs, seq_len, dm)
        return z

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear1 = nn.Linear(cfg.dm, cfg.dff)
        self.act = nn.GELU() if cfg.gelu else nn.ReLU()
        self.linear2 = nn.Linear(cfg.dff, cfg.dm)
        self.dropout = nn.Dropout(cfg.resid_dropout)
    
    def forward(self, x):
        return self.dropout(self.linear2(self.act(self.linear1(x))))        

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(cfg)
        self.ffn = FeedForwardNetwork(cfg)
        self.ln1 = nn.LayerNorm(cfg.dm)
        self.ln2 = nn.LayerNorm(cfg.dm)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Generator(nn.Module):
    pass

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([Block(cfg) for i in range(cfg.n_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
