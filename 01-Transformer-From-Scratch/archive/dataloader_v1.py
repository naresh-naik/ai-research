# Step 1 -Import

from torch.utils.data import DataLoader

from dataset import (
    TranslationDataset,
    training_pairs
)

from vocabulary import Vocabulary

from tokenizer import Tokenizer

# Step 2- Build Vocabulary

vocab = Vocabulary()

vocab.add_special_tokens()

all_sentences = []

for src, tgt in training_pairs:

    all_sentences.append(src)

    all_sentences.append(tgt)

vocab.build_vocabulary(all_sentences)

# Step 3 -Tokenizer

tokenizer = Tokenizer(vocab)


# Step 4 - Dataset

dataset = TranslationDataset(
    training_pairs,
    tokenizer
)

# Step 5 - Create DataLoader

loader = DataLoader(

    dataset,

    batch_size=2,       # give two examples at a time

    shuffle=True        # randomize the order each epoch
    
)


# Step 6-Test Block

print("\nNumber of batches:")

print(len(loader))

print()

for batch in loader:

    print("Encoder Input")

    print(batch["encoder_input"])

    print()

    print("Decoder Input")

    print(batch["decoder_input"])

    print()

    print("Label")

    print(batch["label"])

    print()
    
    print("----------------------------")