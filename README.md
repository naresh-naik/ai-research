# AI Research

Two AI/ML projects that build on each other: implementing a Transformer's
core mechanics by hand, then applying a published mechanistic-interpretability
technique to a real pretrained language model (GPT-2).

```
Transformer fundamentals (from-scratch encoder-decoder)
        ↓
Transformer internals via TransformerLens (GPT-2)
        ↓
Applying Anthropic's Jacobian Lens to GPT-2
```

Each project is self-contained, with its own documentation, dependencies,
and license — see the notes below for exactly how licensing is split.

## Projects

### 01 — Transformer From Scratch

An encoder-decoder Transformer (Vaswani et al., 2017) implemented from
scratch in PyTorch — embeddings, positional encoding, multi-head attention,
encoder/decoder stacks, and greedy decoding, with no `nn.Transformer`
shortcuts. Trained on a small, hand-written English→French dataset (8
training / 1 validation / 1 test examples) — an educational demonstration,
not a benchmark or production translation system.

**Key technologies:** PyTorch, custom tokenizer/vocabulary pipeline, attention
visualization.

**Repository:** [`01-Transformer-From-Scratch/`](01-Transformer-From-Scratch/) ·
[github.com/naresh-naik/transformer-from-scratch](https://github.com/naresh-naik/transformer-from-scratch)

### 02 — GPT-2 Mechanistic Interpretability

Two components: introductory TransformerLens exercises on GPT-2, and a
research project applying Anthropic's **Jacobian Lens** technique
(originally demonstrated on Qwen models) to GPT-2 — including a custom
synthetic prompt corpus, a GPT-2-specific fitted lens, and a Jacobian Lens
vs. logit-lens comparison. The Jacobian Lens algorithm itself is Anthropic's
work; this project's contribution is the GPT-2 application and the
engineering required to get it running.

**Key technologies:** GPT-2 (small), TransformerLens, Hugging Face
Transformers, Anthropic's `jlens` library.

**Repository:** [`02-GPT2-Mechanistic-Interpretability/`](02-GPT2-Mechanistic-Interpretability/) ·
[github.com/naresh-naik/gpt2-mechanistic-interpretability](https://github.com/naresh-naik/gpt2-mechanistic-interpretability)

## What This Repository Demonstrates

- **PyTorch / Deep Learning** — from-scratch model implementation, training
  loops, checkpointing, reproducible pipelines.
- **Transformers & Attention** — multi-head self-attention, cross-attention,
  positional encoding, residual streams, implemented and inspected directly.
- **NLP** — tokenization, vocabulary construction, sequence-to-sequence data
  pipelines.
- **Mechanistic Interpretability** — TransformerLens fundamentals, activation
  caching, Jacobian-based representation analysis, logit-lens comparison.
- **Research Engineering** — adapting and debugging an external open-source
  research codebase (Anthropic's Jacobian Lens) to run against a new model.
- **Debugging** — diagnosing and fixing real execution bugs (a
  batch-dimension bug in Project 01's attention path; a checkpoint-path and
  attribute-name bug in Project 02's analysis notebook).
- **Reproducibility & Engineering Practice** — documented setup, honest
  limitations, license/attribution hygiene, clean version control.

## Repository Structure

```
ai-research/
├── 01-Transformer-From-Scratch/
│   ├── model.py, decoder.py, transformer.py, ...   # Encoder-decoder implementation
│   ├── data/                                        # Tiny English→French dataset
│   ├── archive/                                     # Earlier implementation history
│   └── README.md
│
├── 02-GPT2-Mechanistic-Interpretability/
│   ├── foundational/transformer-lens-learning/       # TransformerLens fundamentals on GPT-2
│   ├── jacobian-lens/
│   │   ├── jlens/                                   # Anthropic's Jacobian Lens library (Apache 2.0)
│   │   ├── notebooks/                               # GPT-2 fitting + analysis notebooks
│   │   ├── english_corpus/                          # Custom synthetic prompt generator
│   │   └── Notes/                                   # Fitting-algorithm study diagrams
│   └── README.md
│
├── README.md      # This file
├── LICENSE         # Per-project licensing overview
├── NOTICE.md        # Per-project attribution overview
└── .gitignore
```

## Project Status

**Completed implementation:**
- Project 01's full encoder-decoder Transformer, with working training,
  inference, and attention-analysis pipelines.
- Project 02's foundational TransformerLens exercises on GPT-2.
- Project 02's GPT-2 Jacobian Lens fitting/application pipeline.

**Completed research demonstration:**
- Project 02's Jacobian Lens vs. logit-lens comparison on GPT-2.
- Structural inspection of a GPT-2-specific fitted lens (`d_model = 768`,
  11 source layers, `[768 × 768]` Jacobian matrices) — fit on a single
  prompt (`n_prompts = 1`), a demonstration-scale artifact, not a
  statistically robust evaluation.

**Planned, not yet implemented:**
- Project 02's deeper analysis roadmap (matrix statistics, heatmaps, SVD,
  layer-wise comparisons, cosine similarity, scaling to 100 prompts,
  stability comparisons, extension to code-focused models).

See each project's own README for full detail and honest limitations.

## Attribution

- **Project 01** contains an original implementation and dataset developed
  for this project. It uses PyTorch and Matplotlib as installed
  dependencies only — no third-party source is vendored.
- **Project 02** incorporates and adapts Anthropic's Jacobian Lens
  implementation (Apache License 2.0, Copyright 2026 Anthropic PBC —
  [github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)).
  The Jacobian Lens algorithm itself is Anthropic's — it was not developed
  in this repository. Project 02 also uses TransformerLens (MIT,
  TransformerLensOrg) as an installed dependency for its foundational
  component. Full attribution detail is in
  [`02-GPT2-Mechanistic-Interpretability/NOTICE.md`](02-GPT2-Mechanistic-Interpretability/NOTICE.md).

## Individual Repositories

- 01 — Transformer From Scratch: <https://github.com/naresh-naik/transformer-from-scratch>
- 02 — GPT-2 Mechanistic Interpretability: <https://github.com/naresh-naik/gpt2-mechanistic-interpretability>

## License

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
