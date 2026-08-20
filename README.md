# AI Research Projects

This repository contains two independent projects. Each is self-contained,
has its own documentation, and its own licensing — see the notes below and
each project's own `README.md`/`LICENSE` for details.

## 01 — Transformer From Scratch

An educational implementation of the original encoder-decoder Transformer
architecture (Vaswani et al., 2017), built from scratch in PyTorch and
trained on a tiny English→French translation dataset.

Path: [`01-Transformer-From-Scratch/`](01-Transformer-From-Scratch/)

See [`01-Transformer-From-Scratch/README.md`](01-Transformer-From-Scratch/README.md)
for the full project documentation, setup, and reproduction instructions.

## 02 — GPT-2 Mechanistic Interpretability

Two related components: introductory exercises with TransformerLens on
GPT-2, and a research project applying Anthropic's Jacobian Lens technique
to GPT-2, including a custom fitting corpus and analysis notebooks.

Path: [`02-GPT2-Mechanistic-Interpretability/`](02-GPT2-Mechanistic-Interpretability/)

See [`02-GPT2-Mechanistic-Interpretability/README.md`](02-GPT2-Mechanistic-Interpretability/README.md)
for the full project documentation, attribution, setup, and reproduction
instructions.

## Licensing

The two projects are licensed independently and are **not** covered by a
single unified license:

- `01-Transformer-From-Scratch/` is MIT-licensed — see its own
  [`LICENSE`](01-Transformer-From-Scratch/LICENSE).
- `02-GPT2-Mechanistic-Interpretability/` is Apache License 2.0 for its own
  original material, and incorporates Anthropic's Apache-2.0-licensed
  Jacobian Lens code and an MIT-licensed TransformerLens dependency — see
  its own [`LICENSE`](02-GPT2-Mechanistic-Interpretability/LICENSE) and
  [`NOTICE.md`](02-GPT2-Mechanistic-Interpretability/NOTICE.md) for the full
  breakdown.

See [`NOTICE.md`](NOTICE.md) at this repository's root for a short pointer
to both.
