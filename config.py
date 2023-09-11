class GPTConfig:
    def __init__(
        self, 
        layer_norm_epsilon=1e-05,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12, 
        n_positions=1024,
        vocab_size=50257,
        gelu=True,
    ):
        self.layer_norm_epsilon = layer_norm_epsilon
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_positions = n_positions
        self.vocab_size = vocab_size
        self.gelu = gelu
