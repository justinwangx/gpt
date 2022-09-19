import torch
import torch.nn as nn
import math

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
        q, k, v = self.qkv_proj(x).split(self.dm, dim=-1) # (bs, seq_len, dm)
        q, k, v = [x.view(bs, seq_len, -1, self.dk).transpose(1, 2) for x in (q, k, v)] # (bs, nh, seq_len, dk)

        w = q @ k.transpose(-2, -1) / math.sqrt(self.dk) # (bs, nh, seq_len, seq_len)
        w.masked_fill_(self.mask[:, :, :seq_len, :seq_len] == 0, -float('inf'))
        w = self.softmax(w)
        w = self.attn_dropout(w)

        z = w @ v # (bs, nh, seq_len, dk)
        z = z.transpose(1, 2).contiguous().view(bs, seq_len, dm)
        z = self.resid_dropout(self.output(z))
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

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_positions = cfg.n_positions
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dm)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.n_positions, cfg.dm))
        self.embed_dropout = nn.Dropout(cfg.embed_dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.dm)
        self.out = nn.Linear(cfg.dm, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        bs, seq_len = x.shape
        assert seq_len <= self.n_positions, 'input contains too many tokens'
        # embed and add positional embedding
        token_embed = self.tok_embed(x)
        pos_embed = self.pos_embed[:, :seq_len, :]
        x = self.embed_dropout(token_embed + pos_embed)
        # decoder blocks
        for layer in self.layers:
            x = layer(x)
        # output
        logits = self.out(self.ln(x))
        return logits
