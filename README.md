# simpleGPT

This is a minimal implementation of GPT-2 in ~100 lines of PyTorch. The code here supports loading GPT-2 weights from HuggingFace and generating text on the CPU. For simplicity, since this is meant to be a simple example of inference and not training, I've omitted some things like dropout and weight initialization.

To run this locally, first clone the repo and then run:

```bash
pip install torch transformers
python generate.py
```

This implementation was inspired by Andrej Karpathy's minGPT / nanoGPT projects. I referenced those when writing this.
