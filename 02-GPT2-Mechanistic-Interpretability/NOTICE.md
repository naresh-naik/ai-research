# Notice

This repository contains four categories of material. Each is described
below, with its license and, where applicable, upstream source.

## 1. User-authored project material

The following are original work created for this project:

- `foundational/transformer-lens-learning/` — 7 introductory scripts written
  while learning the TransformerLens API on GPT-2 (see the top-level README
  for why this is labeled foundational learning, not research).
- `jacobian-lens/english_corpus/` — a custom synthetic English prompt
  generator.
- `jacobian-lens/notebooks/01_GPT2_Walkthrough.ipynb`,
  `jacobian-lens/notebooks/02_Jacobian_Analysis.ipynb`,
  `jacobian-lens/walkthrough_my.ipynb` — GPT-2-specific application,
  fitting, and analysis notebooks.
- `jacobian-lens/Notes/` — mathematical study/annotation diagrams of the
  Jacobian Lens fitting mechanics.
- The comment-only study annotations added inside `jacobian-lens/jlens/`
  (see item 2 below for the boundary).

This material is licensed under the Apache License, Version 2.0 (see
[LICENSE](LICENSE)).

## 2. Anthropic Jacobian Lens code — Apache License 2.0

`jacobian-lens/jlens/` is Anthropic's own Jacobian Lens implementation:

- Upstream repository: <https://github.com/anthropics/jacobian-lens>
- Upstream license: Apache License, Version 2.0
- Upstream copyright: Copyright 2026 Anthropic PBC
- Companion paper: [*Verbalizable Representations Form a Global Workspace in
  Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)

Every file in `jacobian-lens/jlens/` retains Anthropic's original copyright
and SPDX license headers unmodified. The Jacobian Lens algorithm itself —
the averaged-Jacobian transport, the fitting procedure, and the
apply/visualization logic — is Anthropic's work and is **not** original to
this repository.

Six files in `jacobian-lens/jlens/` (`__init__.py`, `fitting.py`, `hf.py`,
`hooks.py`, `lens.py`, `protocol.py`) contain local, comment-only study
annotations (architecture diagrams and explanatory notes written while
learning the codebase). No functional logic in any of these files was
changed. As required by the Apache License, Version 2.0 (§4(b)), this NOTICE
states that these files have been modified from their original upstream
form.

## 3. TransformerLens — MIT-licensed dependency

`foundational/transformer-lens-learning/` uses
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) as
an installed dependency:

- License: MIT
- Copyright: Copyright (c) 2022 TransformerLensOrg

TransformerLens is **not** vendored or redistributed as source in this
repository — it is installed via `pip` per `requirements.txt`. It is used
only by `foundational/transformer-lens-learning/`; the `jacobian-lens/`
component does not use TransformerLens.

## 4. GPT-2 / Hugging Face ecosystem dependencies

GPT-2 (OpenAI, Radford et al., 2019, *Language Models are Unsupervised
Multitask Learners*) is loaded via Hugging Face `transformers` and, in the
foundational component, via TransformerLens's own `from_pretrained` wrapper.
No model weights are bundled with this repository; they are downloaded at
runtime and are subject to their own respective license terms, independent
of this project. `huggingface_hub` and `datasets` (Hugging Face) are used as
installed dependencies only, not redistributed as source.

## License

- Sections 1 above: Apache License, Version 2.0 — see [LICENSE](LICENSE).
- Section 2 above: Apache License, Version 2.0, Copyright 2026 Anthropic PBC
  — governed by the original headers within those files.
- Sections 3–4 above: third-party dependencies, governed by their own
  respective licenses, not redistributed as source by this repository.

This repository does not claim ownership of Anthropic's Jacobian Lens
implementation or of TransformerLens.
