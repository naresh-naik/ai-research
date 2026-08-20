# Import your vocabulary

from vocabulary import Vocabulary, training_sentences


# Step 2 - Tokenizer Class

class Tokenizer:

    def __init__(self, vocabulary):

        self.vocab = vocabulary

    def tokenize(self, sentence):

        return sentence.lower().split()
    
      


# Step 3 - Encode Method

    def encode(self, sentence):

        words = self.tokenize(sentence)

        ids = []

        ids.append(
            self.vocab.word_to_id["<SOS>"]
        )

        for word in words:

            if word in self.vocab.word_to_id:

                ids.append(
                    self.vocab.word_to_id[word]
                )

            else:

                ids.append(
                    self.vocab.word_to_id["<UNK>"]
                )

        ids.append(
            self.vocab.word_to_id["<EOS>"]
        )

        return ids


# Step 4 - Decode Method

    def decode(self, token_ids):

        words = []

        for token_id in token_ids:

            word = self.vocab.id_to_word.get(
                token_id,
                "<UNK>"
            )

            if word in ["<SOS>", "<EOS>", "<PAD>"]:

                continue

            words.append(word)

        return " ".join(words)



# TEST BLOCK

if __name__ == "__main__":

    vocab = Vocabulary()

    vocab.add_special_tokens()

    vocab.build_vocabulary(training_sentences)

    tokenizer = Tokenizer(vocab)

    sentence = "Explain transformers in simple words"

    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.encode(
        sentence
    )

    print("\nOriginal Sentence:")
    print(sentence)

    print("\nTokens:")
    print(tokens)

    print("\nEncoded:")
    print(token_ids)

    decoded = tokenizer.decode(token_ids)

    print("\nDecoded:")
    print(decoded)
