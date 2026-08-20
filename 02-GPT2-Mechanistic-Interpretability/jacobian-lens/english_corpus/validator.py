from collections import Counter


class CorpusValidator:

    def __init__(
        self,
        min_words=60,
        max_words=150
    ):

        self.min_words = min_words
        self.max_words = max_words

    def valid_length(self, paragraph):

        words = paragraph.split()

        return self.min_words <= len(words) <= self.max_words

    def not_empty(self, paragraph):

        return bool(paragraph.strip())

    def english(self, paragraph):

        try:

            paragraph.encode("ascii")

            return True

        except UnicodeEncodeError:

            return False

    def remove_duplicates(self, paragraphs):

        return list(dict.fromkeys(paragraphs))

    def validate(self, paragraphs):

        paragraphs = self.remove_duplicates(paragraphs)

        cleaned = []

        for paragraph in paragraphs:

            if not self.not_empty(paragraph):
                continue

            if not self.english(paragraph):
                continue

            if not self.valid_length(paragraph):
                continue

            cleaned.append(paragraph)

        return cleaned