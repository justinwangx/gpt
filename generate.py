from config import GPTConfig
from gpt import GPT
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_hf_weights(model: GPT):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    state_dict = model.state_dict()
    state_dict_hf = model_hf.state_dict()

    assert(set(state_dict.keys()) == set(state_dict_hf.keys())), "check that architectures match"

    # need to transpose weights since we use nn.Linear
    transpose = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    for k in list(state_dict.keys()):
        if any(s in k for s in transpose):
            state_dict[k] = state_dict_hf[k].T
        else:
            state_dict[k] = state_dict_hf[k]
    model.load_state_dict(state_dict)

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    cfg = GPTConfig()
    model = GPT(cfg)
    load_hf_weights(model)

    prompt = "Hello GPT, how are you doing?"
    input = tokenizer(prompt, return_tensors="pt")
    # we can just take the input ids since the implementation attends to every token (don't need the attention mask)
    input = input["input_ids"]
    output = model.generate(input, top_k=10)
    response = tokenizer.decode(output[0])
    print(response)

if __name__ == "__main__":
    main()



    