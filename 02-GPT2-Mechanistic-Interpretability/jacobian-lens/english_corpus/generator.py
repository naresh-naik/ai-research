import json
import random
from pathlib import Path

from templates import GEOGRAPHY_TEMPLATES


class GeographyGenerator:

    def __init__(self, json_path):
        self.json_path = Path(json_path)

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def generate_one(self):

        country = random.choice(self.data)

        template = random.choice(GEOGRAPHY_TEMPLATES)

        return template.format(**country)

    def generate(self, n=100):

        paragraphs = []

        for _ in range(n):
            paragraphs.append(
                self.generate_one()
            )

        return paragraphs

    def save(self, paragraphs, output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w", encoding="utf-8") as f:

            for paragraph in paragraphs:

                f.write(paragraph)

                f.write("\n\n")