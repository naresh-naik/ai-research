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


prompt = "I love AI"

tokens = model.to_tokens(prompt)

print(tokens)

print(model.to_str_tokens(prompt))


logits, cache = model.run_with_cache(tokens)

print("\nLogits Shape:")

print(logits.shape)

print("\nNumber of Cached Activations:")

print(len(cache))

print("\nThe Cache Keys:")
print(cache.keys())