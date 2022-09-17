class GPTConfig:
    def __init__(
        self, 
        n_layers=12, 
        n_heads=12,
        dm=768,
        dff=3072,
        attn_dropout=0.1,
        resid_dropout=0.1,
        n_positions=1024,
        gelu=True,
    ):

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dm = dm
        self.dff = dff
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.n_positions = n_positions
        self.gelu = gelu
