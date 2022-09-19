from config import GPTConfig
from gpt import GPT

import torch
from torch.nn import functional as F
import torch.optim as optim

def overfit_example_input(vocab, input, n_steps=200, print_loss=False):
    cfg = cfg = GPTConfig(n_layers=2, n_heads=2, vocab_size=len(vocab), dm=128, dff=256, n_positions=512)
    model = GPT(cfg)
    # not using AdamW because the current model doesn't support weight decay for simplicity
    # (would have to exclude layernorm and embedding parameters)
    optimizer = optim.Adam(model.parameters())

    # untrained
    logits = model(input)
    print(f'logits: {logits}')
    preds = torch.argmax(logits, dim=-1).squeeze(0)
    print(f'preds: {preds}')
    for idx in preds:
        print(vocab[idx.item()])

    model.train()
    for _ in range(n_steps):
        logits = model(input)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input.view(-1), ignore_index=-1)
        if print_loss:
            print(f'Step {n_steps+1}. Loss: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # trained
    logits = model(input)
    print(f'logits: {logits}')
    new_preds = torch.argmax(logits, dim=-1).squeeze(0)
    print(f'new preds: {new_preds}')
    for idx in new_preds:
        print(vocab[idx.item()]) 

if __name__ == '__main__':
    vocab = {
        0: 'this',
        1: 'is',
        2: 'an',
        3: 'example',
        4: 'vocabulary'
    }
    input = torch.LongTensor([[0, 1, 2, 3, 4]])
    overfit_example_input(vocab, input)
