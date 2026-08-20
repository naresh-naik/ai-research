# Step 1 - Imports


import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from tokenizer import Tokenizer
from dataset import TranslationDataset
from collate import collate_fn

from transformer import Transformer

# Add Devices Selection

# ==========================
# Device
# ==========================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")

print()

print("=" * 60)

print(f"Running on: {device}")

print("=" * 60)




# Step 2 - Build Vocabulary


vocab = Vocabulary()

vocab.add_special_tokens()

vocab.build_from_file("data/train.txt")

# Step 3 - Tokenizer

tokenizer = Tokenizer(vocab)


# Step 4- Dataset

dataset = TranslationDataset(

    "data/train.txt",

    tokenizer

)

# ==========================
# Validation Dataset
# ==========================

valid_dataset = TranslationDataset(

    "data/valid.txt",

    tokenizer

)

# Step 5 - DataLoader


train_loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True,

    collate_fn=collate_fn

)

# ==========================
# Validation DataLoader
# ==========================

valid_loader = DataLoader(

    valid_dataset,

    batch_size=2,

    shuffle=False,

    collate_fn=collate_fn

)



# Step 6 - Print

print("Vocabulary Size:")

print(len(vocab.word_to_id))

print()

print("Dataset Size:")

print(len(dataset))

print()

print("Number of Batches:")

print(len(train_loader))



# ==========================
# Build Transformer
# ==========================

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

model = model.to(device)  # move the model to the device

print()

print("=" * 60)
print("Transformer Built Successfully!")
print("=" * 60)



# ===========================
# Loss Function
# ===========================

criterion = nn.CrossEntropyLoss(
    ignore_index=0      # Pytorch automatically skips PAD tokens
)

# ==========================
# Optimizer
# ==========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)




# ==========================
# Test One Forward Pass
# ==========================

# batch = next(iter(train_loader))

# encoder_input, decoder_input, labels = batch

# print()

# print("Encoder Shape:")
# print(encoder_input.shape)

# print()

# print("Decoder Shape:")
# print(decoder_input.shape)

# print()

# logits = model(

#     encoder_input,

#     decoder_input

# )
# # ==========================
# # Flatten Logits
# # ==========================

# logits = logits.view(

#     -1,

#     logits.shape[-1]

# )

# labels = labels.view(-1)

# print()

# print("Flattened Logits Shape:")

# print(logits.shape)

# print()

# print("Flattened Labels Shape:")

# print(labels.shape)

# # ==========================
# # Compute Loss
# # ==========================

# loss = criterion(

#     logits,

#     labels

# )

# print()

# print("Loss:")

# print(loss.item())







# # ==========================
# # Clear Old Gradients
# # ==========================

# optimizer.zero_grad()

# # ==========================
# # Backpropagation
# # ==========================

# loss.backward()

# # ==========================
# # Update Weights
# # ==========================

# optimizer.step()

# print()

# print("Gradient Shape:")

# print(
#     model.encoder.layers[0]
#          .self_attention_block
#          .w_q.weight.grad.shape
# )

# print()

# print("Weights Updated Successfully!")


# ==========================
# Best Validation Loss
# ==========================
   

best_valid_loss = float("inf")







# ==========================
# Training Loop
# ==========================

num_epochs = 20

for epoch in range(num_epochs):

    # ==========================
    # Training Mode
    # ==========================

    model.train()

    total_loss = 0



    for encoder_input, decoder_input, labels in train_loader:


        encoder_input = encoder_input.to(device)

        decoder_input = decoder_input.to(device)

        labels = labels.to(device)

        logits = model(
            encoder_input,
            decoder_input
        )

        logits = logits.view(
            -1,
            logits.shape[-1]
        )

        labels = labels.view(-1)

        loss = criterion(
            logits,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)




    print(
        f"Epoch {epoch+1:2d} | Train Loss = {average_loss:.4f}"
    )


    # ==========================
    # Validation
    # ==========================

    model.eval()

    valid_loss = 0

    with torch.no_grad():

        for encoder_input, decoder_input, labels in valid_loader:

            encoder_input = encoder_input.to(device)

            decoder_input = decoder_input.to(device)

            labels = labels.to(device)
            

            logits = model(

                encoder_input,

                decoder_input

            )

            logits = logits.view(

                -1,

                logits.shape[-1]

            )

            labels = labels.view(-1)

            loss = criterion(

                logits,

                labels

            )

            valid_loss += loss.item()

    average_valid_loss = valid_loss / len(valid_loader)

    print(

        f"Validation Loss = {average_valid_loss:.4f}"

    )

   




    # ==========================
    # Save Best Model
    # ==========================

    if average_valid_loss < best_valid_loss:

        best_valid_loss = average_valid_loss

        torch.save(

            model.state_dict(),

            "best_transformer_model.pth"

        )

        print("✅ Best Model Saved!")

    print("-" * 50)

