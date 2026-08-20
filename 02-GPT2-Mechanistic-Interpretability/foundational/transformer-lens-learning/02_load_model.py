import torch
from transformer_lens import HookedTransformer

### Device

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else"cpu"
)

print(f"Device: {device}")


### Load GPT-2

model = HookedTransformer.from_pretrained("gpt2")


### -Print Basic Information

print("\n Model Information")
print("="*50)

print("Model Name:", model.cfg.model_name)

print("Layers:",model.cfg.n_layers)

print("Attention Heads:",model.cfg.n_heads)

print("MLP Dimension:",model.cfg.d_mlp)

print("Vocabulary Size:",model.cfg.d_vocab)

print("Context Length:",model.cfg.n_ctx)

print("Model Dimension:",model.cfg.d_model)

print("Head Dimension:",model.cfg.d_head)


