# Step 1 - Imports

import torch
from torch.utils.data import Dataset

from vocabulary import Vocabulary
from tokenizer import Tokenizer


# Step 2 - Training Data

training_pairs = [
        ("i love ai", "j aime ia"),
        ("good morning", "bonjour"),
        ("thank you", "merci"),
        ("how are you", "comment allez vous")
]


# Step 3 - Dataset Class

class TranslationDataset(Dataset):

        def __init__(self, pairs, tokenizer):

                self.pairs = pairs
                self.tokenizer = tokenizer

        # Step 4 - Length Method

        def __len__(self):

                return len(self.pairs)

        # Step 5 - Get Item Method

        def __getitem__(self, idx):

                source_sentence, target_sentence = self.pairs[idx]

                encoder_input = self.tokenizer.encode(source_sentence)
                target_ids = self.tokenizer.encode(target_sentence)

                decoder_input = target_ids[:-1] # Take everything except the last element
                label = target_ids[1:]      # [1:]-> skip the first element

                return {
                        "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
                        "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
                        "label": torch.tensor(label, dtype=torch.long),
                }


if __name__ == "__main__":

        vocab = Vocabulary()
        vocab.add_special_tokens()

        all_sentences = []
        for src, tgt in training_pairs:
                all_sentences.append(src)
                all_sentences.append(tgt)

        vocab.build_vocabulary(all_sentences)

        tokenizer = Tokenizer(vocab)
        dataset = TranslationDataset(training_pairs, tokenizer)

        print("Dataset Size:")
        print(len(dataset))
        print()

        sample = dataset[0]

        print("Encoder Input:")
        print(sample["encoder_input"])
        print()

        print("Decoder Input:")
        print(sample["decoder_input"])
        print()

        print("Label:")
        print(sample["label"])