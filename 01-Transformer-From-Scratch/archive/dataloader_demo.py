import torch

from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from tokenizer import Tokenizer
from dataset_v2 import TranslationDataset

from collate import collate_fn


# ----------------------------
# Build Vocabulary
# ----------------------------

vocab = Vocabulary()

vocab.add_special_tokens()

vocab.build_from_file("data/train.txt")


# ----------------------------
# Tokenizer
# ----------------------------

tokenizer = Tokenizer(vocab)


# ----------------------------
# Dataset
# ----------------------------

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



# ----------------------------
# DataLoader
# ----------------------------

loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True,

    collate_fn=collate_fn

)


# ----------------------------
# Test
# ----------------------------

print("Number of batches:")

print(len(loader))


for batch in loader:

    encoder_input, decoder_input, labels = batch

    print("\n=========================")
    print("NEW BATCH")
    print("=========================")

    print("\nEncoder Input")
    print(encoder_input)

    print()

    print("Decoder Input")
    print(decoder_input)

    print()

    print("Labels")
    print(labels)

    break