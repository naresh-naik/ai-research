# Step 1 - Training Data
training_sentences = [

    "i love ai",

    "good morning",

    "thank you",

    "how are you",

    "i am happy",

    "good night",

    "Explain transformers in simple words"

]


# Step 2 - Vocabulary Class

class Vocabulary:

    def __init__(self):

        self.word_to_id = {}

        self.id_to_word = {}

        self.next_id = 0

    # Step 3 - Add Special Tokens

    def add_special_tokens(self):

        special_tokens = [

            "<PAD>",

            "<SOS>",

            "<EOS>",

            "<UNK>"

        ]

        for token in special_tokens:

            self.word_to_id[token] = self.next_id

            self.id_to_word[self.next_id] = token

            self.next_id += 1


    def add_word(self, word):

        if word not in self.word_to_id:

            self.word_to_id[word] = self.next_id

            self.id_to_word[self.next_id] = word

            self.next_id += 1


    def build_vocabulary(self, sentences):

        for sentence in sentences:

            words = sentence.lower().split()

            for word in words:

                if word not in self.word_to_id:

                    self.word_to_id[word] = self.next_id

                    self.id_to_word[self.next_id] = word

                    self.next_id += 1

   


    # BUILD Vocabulary (automatic) used in the dataset file

    def build_from_file(self, file_path):

        with open(file_path, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if line == "":
                    continue

                source, target = line.split("|||")

                source_words = source.strip().lower().split()

                target_words = target.strip().lower().split()

                for word in source_words:

                    self.add_word(word)         # called n times (n = sum of source + target tokens)

                for word in target_words:

                    self.add_word(word)



# TEST BLOCK


if __name__ == "__main__":

    vocab = Vocabulary()

    vocab.add_special_tokens()

    vocab.build_from_file("data/train.txt")

    print("\nWord -> ID\n")

    for word, idx in vocab.word_to_id.items():

        print(f"{word:12} -> {idx}")

    print("\nTotal Vocabulary Size:")
    print(len(vocab.word_to_id))