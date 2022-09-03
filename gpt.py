import torch
import torch.nn as nn

def positional_encoding(x):
    pass

class MultiHeadAttention(nn.Module):
    def __init__(self, dm, nh):
        # i'm setting dv = dk = dm / nh, as this is what's done in the original transformer paper
        # this makes the computational cost of MHA similiar to that of attention for a single head
        # but if we wanted to we could set dk and dm to different values
        assert dm % nh == 0, 'dimension of model not divisible by number of heads'
        super().__init__()
        self.dk = dm // nh
        self.dv = self.dk
        self.q_proj = nn.Linear(dm, dm)
        self.k_proj = nn.Linear(dm, dm)
        self.v_proj = nn.Linear(dm, dm)
        self.output = nn.Linear(dm, dm)
    
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
        

class AttentionBlock(nn.Module):
    def __init__(self, n_heads, d_model, ff_dim):
        super.__init__()
        self.MHA = MultiHeadAttention(d_model, n_heads)
        self.FF1 = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.FF2 = nn.Linear(ff_dim, d_model)
        self.LN = nn.LayerNorm()

    def forward(self, x):
        x = x + self.MHA(x)
        x = self.LN(x)
        x = x + self.FF2(self.relu(self.FF1(x)))
        x = self.LN(x)
        return x

class GPT(nn.Module):
    def __init__(self, n_heads=3, n_layers=3, d_model=512, ff_dim=256):
        super().__init__()
        self.layers = nn.Sequential(*[AttentionBlock(n_heads, d_model, ff_dim) for i in range(n_layers)])
    
    def forward(self, x):
        x = positional_encoding(x)
        x = self.layers(x)
        return x

