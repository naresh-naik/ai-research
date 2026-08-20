import torch

from transformer import Transformer

from vocabulary import Vocabulary

from tokenizer import Tokenizer


# Step 2

# ----------------------------
# Build Vocabulary
# ----------------------------

vocab = Vocabulary()

vocab.add_special_tokens()

vocab.build_from_file("data/train.txt")

tokenizer = Tokenizer(vocab)


# Step 3

# ==============
# Build The Transformer
# ================


model = Transformer(

    src_vocab_size=len(vocab.word_to_id),

    tgt_vocab_size=len(vocab.word_to_id),

    src_seq_len=20,

    tgt_seq_len=20,

    d_model=512,

    N=6,

    h=8,

    dropout=0.1,

    d_ff=2048,

)


# step 4- Load The Trained Weights


model.load_state_dict(

    torch.load("best_transformer_model.pth", map_location="cpu")

)

model.eval()

print("Model Loaded Successfully!")


# Step 5 - Test Sentence

sentence = "i love ai"

encoder_input = tokenizer.encode(sentence)

encoder_input = torch.tensor(

    encoder_input

).unsqueeze(0)

print()

print("Encoder Input:")

print(encoder_input)


# ==========================
# Encoder Forward Pass
# ==========================

with torch.no_grad():

    encoder_output = model.encode(

        encoder_input

    )

print()

print("Encoder Output Shape:")

print(encoder_output.shape)



# =======================
# Start Decoder--Initilize Decoder Input
# ======================

decoder_input = torch.tensor(

        [[vocab.word_to_id["<SOS>"]]]

)

print()

print("Initial Decoder Input:")

print(decoder_input)


# ====================================
# One Decoder Step---Run Decoder once
# ====================================

with torch.no_grad():

    decoder_output = model.decode(

        encoder_output,

        None,

        decoder_input,

        None
    )

    print()

    print("Decoder Output Shape:")

    print(decoder_output.shape)


# Convert to Vocbulary Scores

    with torch.no_grad():

        logits = model.project(

            decoder_output
        )

    print()

    print("Logits Shape:")

    print(logits.shape)

# Pick the Best Word

    next_token = torch.argmax(

        logits[:,-1],

        dim=-1
    )

    print()

    print("Next Token ID:")

    print(next_token)




# ==========================
# Greedy Decoding
# ==========================

decoder_input = torch.tensor(
    [[vocab.word_to_id["<SOS>"]]]
)

generated_tokens = []

max_length = 20

with torch.no_grad():

    for step in range(max_length):

        decoder_output = model.decode(
            encoder_output,
            None,
            decoder_input,
            None
        )

        logits = model.project(
            decoder_output
        )

        next_token = torch.argmax(
            logits[:, -1],
            dim=-1
        )

        token_id = next_token.item()

        generated_tokens.append(token_id)

        print(
            f"Step {step+1:2d} -> "
            f"{vocab.id_to_word[token_id]}"
        )

        if token_id == vocab.word_to_id["<EOS>"]:
            break

        decoder_input = torch.cat(
            [
                decoder_input,
                next_token.unsqueeze(0)
            ],
            dim=1
        )


    # Decode the generated sentence

    print()

    print("Generated Token IDs:")

    print(generated_tokens)

    print()

    print("Translation:")

    print(
        tokenizer.decode(generated_tokens)
    )


    