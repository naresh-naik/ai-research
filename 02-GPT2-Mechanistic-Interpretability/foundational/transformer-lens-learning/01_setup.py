import torch

from transformer_lens import HookedTransformer


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

model = HookedTransformer.from_pretrained("gpt2")

print("Model Loaded Successfully!")