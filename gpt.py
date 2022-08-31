import torch
import torch.nn as nn

def positional_encoding(x):
    pass

class MultiHeadAttention(nn.Module):
    def __init__(self, dm, nh):
        assert dm % nh == 0, 'dimension of model not divisible by number of heads'
        super.__init__()
        self.dk = dm / nh
        self.dv = self.dk
        self.q_proj = nn.Linear(dm, dm)
        self.k_proj = nn.Linear(dm, dm)
        self.v_proj = nn.Linear(dm, dm)
    
    def forward(self, x):
        # x -> (batch_size, seq_len, embed_dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        

class AttentionBlock(nn.Module):
    def __init__(self, n_heads, embed_dim, ff_dim):
        super.__init__()
        self.MHA = MultiHeadAttention(n_heads, embed_dim)
        self.FF1 = nn.Linear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.FF2 = nn.Linear(ff_dim, embed_dim)
        self.LN = nn.LayerNorm()

    def forward(self, x):
        x = x + self.MHA(x)
        x = self.LN(x)
        x = x + self.FF2(self.relu(self.FF1(x)))
        x = self.LN(x)
        return x

class GPT(nn.Module):
    def __init__(self, n_heads=3, n_layers=3, embed_dim=256, ff_dim=256):
        super().__init__()
        self.layers = nn.Sequential(*[AttentionBlock() for i in range(n_layers)])
    
    def forward(self, x):
        x = positional_encoding(x)
        x = self.layers(x)

