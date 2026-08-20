# # This is an important software engineering principle:

# # train.py → trains the model.
# # inference.py → generates translations.
# # attention_analysis.py → inspects the model.

# # Keeping analysis separate from training is how research codebases stay manageable.


# What attention_analysis.py will do

# It will:

# Load the trained model.
# Run a sentence through the encoder.
# Read:
# model.encoder.layers[0].self_attention_block.attention_weights
# Print:
# Shape:
# torch.Size([1, 8, 5, 5])

# Then we'll inspect individual heads:

# Head 0

# [[0.82 0.05 0.04 0.05 0.04]
#  ...
# ]

# Finally, we'll turn those matrices into heatmaps.




import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


from vocabulary import Vocabulary
from tokenizer import Tokenizer
from transformer import Transformer



# Build vocab

vocab =  Vocabulary()

vocab.add_special_tokens()

vocab.build_from_file("data/train.txt")

Tokenizer = Tokenizer(vocab)


# Build transformer exactly like inference.

model = Transformer(
    src_vocab_size=len(vocab.word_to_id),
    tgt_vocab_size=len(vocab.word_to_id),
    src_seq_len=20,
    tgt_seq_len=20,
    d_model=512,
    N=6,
    h=8,
    dropout=0.1,
    d_ff=2048

)

# Load model
model.load_state_dict(

    torch.load(

        "best_transformer_model.pth",

        map_location="cpu"

    )

)

model.eval() # eval() for no randomness and stable experiment

print("Model Loaded Successfully!")


sentence = " i love ai"


# Encode
encoder_input = Tokenizer.encode(sentence)

encoder_input = torch.tensor(

    encoder_input

).unsqueeze(0)


# ==========================
# Forward Pass Through Encoder
# ==========================

with torch.no_grad():

    encoder_output = model.encode(
        encoder_input
    )

print()

print("Number of Hidden States:")
print(len(model.encoder.hidden_states))

print()

for i, hidden in enumerate(model.encoder.hidden_states):

    print(f"Layer {i+1}")

    print(hidden.shape)

    print()

print()

print("Encoder Output Shape:")

print(encoder_output.shape)


# Access Layer 1 Attention

#  =========================
# Get Attention Matrix
# ==========================

# ==========================
# Get Attention Matrix
# ==========================

attention = (
    model.encoder.layers[0]
         .self_attention_block
         .attention_weights
)

print()

print("Attention Shape:")
print(attention.shape)


# Print One Head

print()

print("==========================")
print("Layer 1 - Head 0")
print("==========================")

print(

    attention[0,0] # first sentence and first attention head

)

print()
print("=" * 50)
print("Representation Similarity (LOVE Token)")
print("=" * 50)

love_index = 2

for layer in range(5):

    vec1 = model.encoder.hidden_states[layer][0, love_index]

    vec2 = model.encoder.hidden_states[layer + 1][0, love_index]

    similarity = F.cosine_similarity(
        vec1.unsqueeze(0),
        vec2.unsqueeze(0)
    )

    print(
        f"Layer {layer+1} → Layer {layer+2}: "
        f"{similarity.item():.4f}"
    )



# ==========================
# Plot All Heads Heatmap
# ==========================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16,8)
)

tokens = ["<SOS>", "I", "LOVE", "AI", "<EOS>"]

for head in range(8):

    row = head // 4
    col = head % 4

    ax = axes[row][col]

    ax.imshow(
        attention[0, head]
    )

    ax.set_title(f"Head {head}")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))

    ax.set_xticklabels(
        tokens,
        rotation=45
    )

    ax.set_yticklabels(tokens)

plt.tight_layout()

plt.show()