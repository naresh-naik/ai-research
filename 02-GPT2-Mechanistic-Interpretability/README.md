# GPT-2 Mechanistic Interpretability

## Overview

This repository studies GPT-2's internal representations and contains two
related but distinct components: an introductory TransformerLens exercise
set, and a research project applying Anthropic's **Jacobian Lens** technique
to GPT-2. The two are kept clearly separate throughout this README and the
repository structure, since they represent very different levels of depth
and should not be conflated.

## Results at a Glance

- Model: **GPT-2 small** (`d_model = 768`)
- Fitted lens covers **11 source layers** (layers 0–10)
- Each stored Jacobian matrix: **[768 × 768]**
- **Experiment 1** (lens inspection, in `02_Jacobian_Analysis.ipynb`)
  **executes successfully**
- The fitted lens used for Experiment 1 was fit on **`n_prompts = 1`** — a
  single-prompt, demonstration-scale artifact, not a statistically robust
  evaluation
- Steps 2–9 of the research roadmap (matrix statistics, heatmaps, SVD, layer
  comparisons, cosine similarity, scaling, stability, code models) are
  **planned, not yet implemented**

See [Results / Verified Observations](#results--verified-observations) and
[Current Research Status](#current-research-status) for full detail.

## Project Components

1. **`foundational/transformer-lens-learning/`** — seven short scripts
   written while learning the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
   API on GPT-2 (loading the model, inspecting its config, tokenization,
   embeddings, a cached forward pass, and attention-pattern inspection).
   **This is introductory/foundational learning material, not original
   research.** It demonstrates familiarity with the library and basic
   transformer mechanics, nothing more.

2. **`jacobian-lens/`** — the substantive research work: applying
   Anthropic's publicly released Jacobian Lens implementation to GPT-2,
   including a custom fitting corpus, GPT-2-specific fitted lenses,
   analysis notebooks, and a Jacobian Lens vs. logit lens comparison. The
   Jacobian Lens algorithm itself is Anthropic's — see
   [Attribution](#attribution).

## Foundational TransformerLens Learning

`foundational/transformer-lens-learning/` contains:

| Script | What it does |
|---|---|
| `01_setup.py` | Loads GPT-2 via `HookedTransformer.from_pretrained`, confirms the model loads |
| `02_load_model.py` | Loads GPT-2, prints its architecture config (layers, heads, `d_model`, etc.) |
| `03_tokenization.py` | Demonstrates tokenization via `model.to_tokens`/`to_str_tokens` |
| `04_embeddings.py` | Computes token embeddings, prints their shape |
| `05_positional_embeddings.py` | Adds positional embeddings, constructs the residual stream |
| `06_first_forward_pass.py` | Runs a forward pass with activation caching (`run_with_cache`) |
| `07_attention_patterns.py` | Extracts and prints a layer-0 attention pattern |

These are sequential, tutorial-level exercises with no novel findings — they
exist to demonstrate working familiarity with TransformerLens fundamentals
before the actual research work below.

## Jacobian Lens Research

`jacobian-lens/` applies Anthropic's Jacobian Lens technique to GPT-2. What
was actually done in this repository, distinct from Anthropic's original
implementation:

- Applying the lens specifically to **GPT-2** (Anthropic's own examples use
  Qwen models).
- Building a custom synthetic English prompt corpus (`english_corpus/`) used
  as fitting input.
- Running the GPT-2-specific lens-fitting workflow and producing a fitted
  lens.
- Three notebooks documenting the GPT-2 application, a structural analysis
  experiment, and a Jacobian Lens vs. logit lens comparison.
- A written mathematical study of the fitting mechanics, captured as diagrams
  in `Notes/`.
- A research roadmap for further analysis (partially implemented — see
  [Current Research Status](#current-research-status)).

## Engineering Contribution

Beyond running the existing tool as documented, getting a working GPT-2
pipeline required real adaptation and debugging work:

- Wired GPT-2 into `jlens.from_hf`, a path Anthropic's own examples don't
  cover (their walkthrough targets Qwen models).
- Built `english_corpus/` from scratch — a synthetic prompt generator — to
  provide fitting input independent of Anthropic's bundled examples.
- Diagnosed and fixed two execution bugs while adapting
  `02_Jacobian_Analysis.ipynb`: a checkpoint filename mismatch that raised
  `FileNotFoundError`, and an incorrect attribute reference
  (`lens.jacobian` instead of the library's actual `lens.jacobians`) that
  raised `AttributeError` even after the filename was corrected. Both were
  root-caused and fixed to get Experiment 1 executing end-to-end.
- Read and annotated Anthropic's `jlens` source (see `Notes/` and the
  in-code comments) to understand the fitting/transport math well enough to
  apply it correctly to a new model.

## Technical Background

- **GPT-2**: a decoder-only transformer language model (this project uses
  GPT-2 small: 124M parameters, `d_model=768`, 12 layers).
- **Residual stream**: the running sum of contributions from each layer that
  a transformer's activations flow through.
- **Jacobian Lens**: linearly transports a residual-stream vector from an
  early layer into the final-layer basis using the averaged input-output
  Jacobian `J_l = E[∂h_final/∂h_l]`, then decodes it with the model's own
  unembedding matrix into a ranked list of vocabulary tokens.
- **Logit lens**: the simpler baseline of applying the final-layer
  unembedding directly to an intermediate layer's residual stream, without
  the Jacobian transport. The notebooks compare Jacobian Lens output against
  this baseline.
- **Vocabulary-space projection**: decoding a `d_model`-dimensional
  residual-stream vector into a distribution over the vocabulary via the
  unembedding matrix.
- **Jacobian matrices**: per source layer, a `[d_model, d_model]` matrix
  approximating how a perturbation at that layer propagates to the final
  layer, estimated by averaging gradients over a prompt corpus.

## Repository Structure

```
02-GPT2-Mechanistic-Interpretability/
├── README.md
├── LICENSE
├── NOTICE.md
├── requirements.txt
├── .gitignore
│
├── foundational/
│   └── transformer-lens-learning/
│       ├── 01_setup.py
│       ├── 02_load_model.py
│       ├── 03_tokenization.py
│       ├── 04_embeddings.py
│       ├── 05_positional_embeddings.py
│       ├── 06_first_forward_pass.py
│       └── 07_attention_patterns.py
│
└── jacobian-lens/
    ├── jlens/                          # Anthropic's Jacobian Lens library (Apache 2.0)
    ├── notebooks/
    │   ├── 01_GPT2_Walkthrough.ipynb   # GPT-2 fitting workflow using the custom corpus
    │   └── 02_Jacobian_Analysis.ipynb  # Experiment 1 (completed) + planned further analysis
    ├── english_corpus/                 # Original synthetic English prompt generator
    ├── Notes/
    │   ├── fitting.py visual note.png            # fitting.py — full flow and logic
    │   ├── fitting.py files fit() algo.png        # fit() — averaging algorithm, cost analysis
    │   └── jacobian_prompts() algo visual.png     # jacobian_for_prompt() — per-prompt mechanics
    └── walkthrough_my.ipynb            # Most complete end-to-end GPT-2 demonstration
```

Model checkpoints, the Hugging Face model cache, and generated visualization
outputs are intentionally **excluded from Git** (see `.gitignore`) — they
are either regenerable by running the notebooks/scripts or are local-only
caches.

**Execution order note**: `02_Jacobian_Analysis.ipynb`'s Experiment 1 loads a
GPT-2 Jacobian Lens checkpoint that is generated by
`01_GPT2_Walkthrough.ipynb`'s fitting workflow. This checkpoint is
intentionally excluded from Git (model/checkpoint artifacts are ignored by
`.gitignore`), so on a fresh clone it will not exist yet. Run
`01_GPT2_Walkthrough.ipynb` first — its fitting cells generate the
checkpoint locally — before running `02_Jacobian_Analysis.ipynb`.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Foundational scripts** (run individually):
```bash
python foundational/transformer-lens-learning/01_setup.py
# ... through 07_attention_patterns.py
```

**Jacobian Lens notebooks**:
```bash
jupyter notebook
```
Then open `jacobian-lens/notebooks/01_GPT2_Walkthrough.ipynb`,
`jacobian-lens/notebooks/02_Jacobian_Analysis.ipynb`, or
`jacobian-lens/walkthrough_my.ipynb` from the Jupyter interface.

GPT-2's model weights are **not stored in this repository** — both
components download them automatically via TransformerLens / Hugging Face
`transformers` on first use. Fitted Jacobian Lens checkpoints are likewise
excluded from Git; running the fitting cells in `01_GPT2_Walkthrough.ipynb`
or `walkthrough_my.ipynb` regenerates them locally.

## Results / Verified Observations

Verified, reproducible facts from the Jacobian Lens component (summarized
above in [Results at a Glance](#results-at-a-glance)):

- Model: **GPT-2 small**, `d_model = 768`.
- The fitted lens used in Experiment 1 covers **11 source layers**
  (layers 0–10), each stored as a `[768, 768]` Jacobian matrix.
- `02_Jacobian_Analysis.ipynb`'s Experiment 1 **executes successfully**,
  loading this fitted lens and printing a structural summary (d_model,
  layer count, per-layer Jacobian shape and dtype).
- The Jacobian Lens vs. logit lens comparison (`lens.apply(...)`) is
  demonstrated successfully in `01_GPT2_Walkthrough.ipynb` and
  `walkthrough_my.ipynb`.

As noted above, this fit used a single prompt (`n_prompts = 1`) and is
demonstration-scale, not a statistically robust evaluation. No accuracy,
benchmark, or performance numbers are reported or claimed anywhere in this
project.

## Current Research Status

**Completed:**
- GPT-2 walkthrough (model loading, corpus loading, fitting workflow)
- Jacobian Lens fitting and application workflow, applied to GPT-2
- Experiment 1 (lens inspection) in `02_Jacobian_Analysis.ipynb`

**Planned, not implemented:**
- Matrix statistics
- Heatmaps
- Singular value decomposition (SVD)
- Layer-wise comparisons
- Cosine similarity analysis
- Scaling the fit to 100 prompts
- Stability comparisons across fits
- Extension to code-focused models

## Limitations

- The foundational TransformerLens scripts (`foundational/`) are
  introductory exercises, not a research contribution — see
  [Foundational TransformerLens Learning](#foundational-transformerlens-learning).
- Results are specific to GPT-2 small and the particular fitted-lens
  configuration used here; they should not be generalized to other models
  or larger prompt sets.
- The Jacobian Lens research roadmap remains partially implemented — see
  [Current Research Status](#current-research-status) for exactly what is
  and isn't done.

## Attribution

**Jacobian Lens** (`jacobian-lens/jlens/`):
- Upstream repository: <https://github.com/anthropics/jacobian-lens>
- Companion paper: [*Verbalizable Representations Form a Global Workspace in
  Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
- License: Apache License, Version 2.0, Copyright 2026 Anthropic PBC
- The Jacobian Lens algorithm is attributed entirely to Anthropic — it was
  not developed in this repository. See [NOTICE.md](NOTICE.md) for the full
  breakdown of upstream vs. local material.

**TransformerLens** (`foundational/transformer-lens-learning/`):
- Repository: <https://github.com/TransformerLensOrg/TransformerLens>
- License: MIT, Copyright (c) 2022 TransformerLensOrg
- Used as an installed dependency only; no TransformerLens source is
  redistributed in this repository.

**GPT-2**: credited to OpenAI (Radford et al., 2019, *Language Models are
Unsupervised Multitask Learners*), loaded via Hugging Face `transformers`
and/or TransformerLens.

## License

- This repository's own original material (`foundational/` and the original
  additions inside `jacobian-lens/`) is licensed under the Apache License,
  Version 2.0 — see [LICENSE](LICENSE).
- `jacobian-lens/jlens/` remains Anthropic's Apache License 2.0 code,
  governed by its own intact copyright/SPDX headers.
- TransformerLens, Hugging Face `transformers`/`datasets`/`huggingface_hub`,
  PyTorch, and NumPy are third-party dependencies governed by their own
  respective licenses and are not redistributed as source by this
  repository.
- Full attribution details are in [NOTICE.md](NOTICE.md).
