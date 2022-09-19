# simple GPT
A simple (~100 lines) implementation of GPT in PyTorch.  
This implementation is based off of the GPT-2 architecture, though some aspects of the model (e.g. scaling the weights of residual layers at initialization) are left out for simplicity.  

```checks.py``` contains a function that checks that the model can learn.  
I may add training / sequence generation scripts at some point in time.

### Implementations / resources that helped me: 
* http://nlp.seas.harvard.edu/annotated-transformer/
* https://github.com/karpathy/minGPT
* https://jalammar.github.io/illustrated-gpt2/