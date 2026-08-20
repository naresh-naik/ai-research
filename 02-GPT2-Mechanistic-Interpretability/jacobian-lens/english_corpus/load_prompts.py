from pathlib import Path

def load_english_prompts(path, n_prompts=None):
    with open(path, "r", encoding="utf-8") as f:
        prompts = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if n_prompts is not None:
        prompts = prompts[:n_prompts]

    return prompts