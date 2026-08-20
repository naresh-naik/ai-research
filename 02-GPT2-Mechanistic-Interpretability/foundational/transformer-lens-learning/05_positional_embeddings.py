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

token_embeddings = model.embed(tokens)

print("\nToken Embedding Shape:")
print(token_embeddings.shape)


positions = torch.arange(tokens.shape[1], device=device)

print("\nPositions:")
print(positions)


position_embeddings = model.pos_embed(tokens)

print("\nPosition Embedding Shape:")
print(position_embeddings.shape)



residual_stream = token_embeddings + position_embeddings

print("\nResidual Stream Shape:")
print(residual_stream.shape)




print("\nToken embedding (first 10 values)")
print(token_embeddings[0,1,:10])

print("\nPosition embedding (first 10 values)")
print(position_embeddings[0,1,:10])

print("\nResidual stream (first 10 values)")
print(residual_stream[0,1,:10])