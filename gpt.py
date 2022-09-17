import torch
import torch.nn as nn

# class PositionalEmbedding(nn.Module):
#     """GPT uses learned positional embeddings"""
#     def __init__(self, cfg):
#         super().__init__()

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        assert cfg.dm % cfg.n_heads == 0, 'dimension of model not divisible by number of heads'
        # i'm setting dv = dk = dm / nh, as this is what's done in the original transformer paper
        # this makes the computational cost of MHA similiar to that of attention for a single head
        # but if we wanted to we could set dk and dm to different values
        super().__init__()
        self.dm = cfg.dm
        self.dk = cfg.dm // cfg.n_heads
        self.dv = self.dk
        self.q_proj = nn.Linear(self.dm, self.dm)
        self.k_proj = nn.Linear(self.dm, self.dm)
        self.v_proj = nn.Linear(self.dm, self.dm)
        self.output = nn.Linear(self.dm, self.dm)
    
    def forward(self, x):
        # x -> (batch_size, seq_len, d_model)
        # Q, K, V -> (batch_size, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # split into heads
        q_split = torch.split(Q, self.dk, dim=2)
        k_split = torch.split(K, self.dk, dim=2)
        v_split = torch.split(V, self.dk, dim=2)
        # att computation for each head
        heads = [torch.softmax((q @ torch.transpose(k, 1, 2)), dim=0) @ v for q, k, v in zip(q_split, k_split, v_split)]
        # concat, multiply by output matrix, return
        return self.output(torch.cat([h for h in heads], dim=2))
        

class Block(nn.Module):
    def __init__(self, cfg):
        super.__init__()
        self.MHA = MultiHeadAttention(cfg)
        self.FF1 = nn.Linear(cfg.dm, cfg.dff)
        self.relu = nn.ReLU()
        self.FF2 = nn.Linear(cfg.dff, cfg.dm)
        self.LN = nn.LayerNorm()

    def forward(self, x):
        x = x + self.MHA(x)
        x = self.LN(x)
        x = x + self.FF2(self.relu(self.FF1(x)))
        x = self.LN(x)
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([Block(cfg) for i in range(cfg.n_layers)])
    
    def forward(self, x):
        x = self.layers(x)
        return x
