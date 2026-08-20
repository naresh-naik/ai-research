import torch
from transformer_lens import HookedTransformer

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = HookedTransformer.from_pretrained("gpt2")



prompt  = "I love AI"

tokens = model.to_tokens(prompt)

print(model.to_str_tokens(prompt))

logits, cache = model.run_with_cache(tokens)


# Retrive Attention Pattern
attention = cache["blocks.0.attn.hook_pattern"]

print(attention.shape)

# print one head

head0 = attention[0,0] # query token and key token (4,4)

print(head0)

print(head0.shape)

attention = cache["blocks.0.attn.hook_pattern"]

print("Attention Shape:")
print(attention.shape)

head0 = attention[0,0]

print("\nHead 0 Shape:")
print(head0.shape)

print("\nHead 0 Attention Matrix:")
print(head0)