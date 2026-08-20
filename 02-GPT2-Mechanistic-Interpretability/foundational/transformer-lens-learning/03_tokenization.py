import torch
from transformer_lens import HookedTransformer

device =(
    "mps"
    if torch.backends.mps.is_available
    else "cuda"
    if torch.cuda.is_available
    else "cpu"
)

model = HookedTransformer.from_pretrained("gpt2")

prompt = "I love AI"

tokens = model.to_tokens(prompt)

print("Tokens:")
print(tokens)

print(tokens.shape)

print(tokens)

print()

print("Shape:",tokens.shape)


print()

print(model.to_str_tokens(prompt))

