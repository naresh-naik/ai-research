import torch

from torch.utils.data import Dataset

from tokenizer import Tokenizer




class TranslationDataset(Dataset):

    def __init__(self, file_path, tokenizer):

        self.samples = []

        self.tokenizer = tokenizer

        with open(file_path, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if line == "":
                    continue

                source, target = line.split("|||")

                source = source.strip()

                target = target.strip()

                self.samples.append((source, target))

    def __len__(self):

        return len(self.samples)

        
    def __getitem__(self, index):

        source_sentence, target_sentence = self.samples[index]

        # ----------------------------
        # Encoder Input
        # ----------------------------

        encoder_input = self.tokenizer.encode(
            source_sentence
        )

        # ----------------------------
        # Target IDs
        # ----------------------------

        target_ids = self.tokenizer.encode(
            target_sentence
        )

        # ----------------------------
        # Decoder Input
        # ----------------------------

        decoder_input = target_ids[:-1]

        # ----------------------------
        # Labels
        # ----------------------------

        labels = target_ids[1:]

        return (

            torch.tensor(encoder_input),

            torch.tensor(decoder_input),

            torch.tensor(labels)

        )


# ===========================
# TEST BLOCK
# ===========================

if __name__ == "__main__":

    from vocabulary import Vocabulary

    vocab = Vocabulary()

    vocab.add_special_tokens()

    vocab.build_from_file(
        "data/train.txt"
    )

    tokenizer = Tokenizer(vocab)

    dataset = TranslationDataset(

        "data/train.txt",

        tokenizer

    )

    print("Number of sentence pairs:")
    print(len(dataset))

    print()

    encoder_input, decoder_input, labels = dataset[0]

    print("Encoder Input:")
    print(encoder_input)

    print()

    print("Decoder Input:")
    print(decoder_input)

    print()

    print("Labels:")
    print(labels)