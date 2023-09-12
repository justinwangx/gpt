from dataclasses import dataclass

@dataclass
class GPTConfig:
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    vocab_size: int = 50257