import torch
import torch.nn as nn
import math

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "embedding dimension must be divisible by number of heads"
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.d_model = cfg.n_embd
        self.n_head = cfg.n_head
        # head dimension dh == dk == dv
        self.dh = self.d_model // self.n_head
        self.mask = torch.tril(torch.ones(1, 1, cfg.n_ctx, cfg.n_ctx))
    
    def forward(self, x):
        bs, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, -1)
        q = q.view(bs, seq_len, self.n_head, self.dh).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_head, self.dh).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_head, self.dh).transpose(1, 2)

        attn = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(self.dh))
        attn = attn.masked_fill(self.mask[..., :seq_len, :seq_len] == 0, -float('inf'))
        attn = nn.functional.softmax(attn, dim=-1)

        z = attn @ v # (bs, nh, seq_len, dh)
        z = z.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        z = self.c_proj(z)
        return z

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4*cfg.n_embd)
        self.c_proj = nn.Linear(4*cfg.n_embd, cfg.n_embd)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class GPT2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MaskedMultiHeadAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = FeedForwardNetwork(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd),
            h = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, x):
        bs, seq_len = x.shape
        assert seq_len <= self.cfg.n_ctx, 'input contains too many tokens'
        pos = torch.arange(seq_len, dtype=torch.long)

        token_embd = self.transformer["wte"](x)
        pos_embd = self.transformer["wpe"](pos)

        x = token_embd + pos_embd
        for layer in self.transformer["h"]:
            x = layer(x)
        
        x = self.transformer.ln_f(x)
        # inference optimization from nanoGPT :) only forward last index to lm head
        logits = self.lm_head(x[:, [-1], :])
        return logits
    
    @torch.no_grad()
    def generate(self, token_idx, max_new_tokens=200, temperature=1.0, top_k=50, do_sample=True):
        for _ in range(max_new_tokens):
            token_idx = token_idx if token_idx.size(1) <= self.cfg.n_ctx else token_idx[:, -self.cfg.n_ctx:]
            logits = self(token_idx)

            # remove sequence dimension (pluck last index) and do temperature scaling
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # mask logits with values below the kth highest value - index with [-1] to preserve the batch dimension
                logits[logits < values[:, [-1]]] = -float('inf')

            probs = nn.functional.softmax(logits, dim=-1)
            if do_sample:
                next_idx = torch.multinomial(probs, num_samples=1)
            else:
                next_idx = torch.argmax(probs, dim=-1, keepdim=True)
            
            token_idx = torch.cat([token_idx, next_idx], dim=1)

        return token_idx
