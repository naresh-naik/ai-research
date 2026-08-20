# Transformer From Scratch

## Overview

This project implements the original encoder-decoder Transformer architecture (Vaswani et al., 2017) from scratch in PyTorch, rather than using a high-level Transformer implementation such as `torch.nn.Transformer` or a pretrained model from Hugging Face. Every core component — embeddings, positional encoding, multi-head attention, the encoder/decoder stacks, and greedy decoding — is implemented manually to build an internal understanding of how a Transformer actually works.

The model is trained on a tiny, hand-written English→French translation dataset. This is **not** a production translation system and does not claim state-of-the-art (or even competitive) translation quality — it is a learning project focused on correctly implementing and inspecting the architecture.

## What This Project Demonstrates

- Token embeddings (`InputEmbeddings`, scaled by √d_model)
- Sinusoidal positional encoding
- Multi-head self-attention (encoder side)
- Masked (causal) multi-head self-attention (decoder side)
- Encoder-decoder cross-attention
- Position-wise feed-forward networks
- Residual connections
- Layer normalization
- A stacked encoder and decoder (configurable number of layers)
- Greedy autoregressive decoding at inference time
- Attention-weight inspection (encoder self-attention, per head)
- Layer-wise hidden-state (representation) analysis via cosine similarity

## Architecture

```
Input Sentence
      │
      ▼
 Tokenization
      │
      ▼
 Token Embeddings + Sinusoidal Positional Encoding
      │
      ▼
┌─────────────────────────┐
│        Encoder ×N        │
│  Multi-Head Self-Attn    │
│  Add & Norm               │
│  Feed Forward              │
│  Add & Norm                │
└─────────────────────────┘
      │
      ▼  (encoder output)
┌─────────────────────────┐
│        Decoder ×N        │
│  Masked Self-Attention    │
│  Add & Norm                │
│  Cross-Attention (enc)      │
│  Add & Norm                  │
│  Feed Forward                 │
│  Add & Norm                    │
└─────────────────────────┘
      │
      ▼
 Linear Projection → Softmax → Next-Token Prediction
```

This mirrors the flow described in [`roadmap.md`](roadmap.md), which documents the project's build philosophy: implement the paper piece by piece, print shapes, visualize, then move to the next component.

## Repository Structure

```
01-Transformer-From-Scratch/
├── model.py                 # Encoder building blocks: embeddings, positional encoding,
│                             #   multi-head attention, feed-forward, encoder stack
├── decoder.py                # Decoder building blocks: masked self-attention,
│                             #   cross-attention, decoder stack, output projection
├── transformer.py            # Assembles model.py + decoder.py into the full
│                             #   encoder-decoder Transformer
├── vocabulary.py              # Whitespace-based vocabulary builder (+ special tokens)
├── tokenizer.py                # Encodes/decodes sentences using the vocabulary
├── dataset.py                   # PyTorch Dataset reading the pipe-delimited data files
├── collate.py                    # Batch padding / collate function for the DataLoader
├── train.py                       # Canonical training + validation entry point
├── inference.py                    # Loads a trained checkpoint and greedy-decodes a sentence
├── attention_analysis.py            # Loads a trained checkpoint and visualizes encoder attention
├── data/                              # train.txt / valid.txt / test.txt (see Dataset below)
├── archive/                            # Earlier v1 implementation and standalone demo scripts,
│                                       #   kept for learning/history — not part of the canonical
│                                       #   pipeline and not required to run the project
└── roadmap.md                          # Original build plan / architecture notes
```

## Dataset

The dataset is intentionally tiny and hand-written, for educational purposes only:

- **10** total English→French sentence pairs
- **8** used for training (`data/train.txt`)
- **1** used for validation (`data/valid.txt`)
- **1** used for testing (`data/test.txt`)

This is **not a benchmark** — with 8 training examples, no meaningful claim about translation quality or generalization can be made. The vocabulary is built only from the training set, so some words appearing only in the validation/test sentences are out-of-vocabulary and map to `<UNK>` at encode time. This is expected, realistic held-out behavior for a genuine (if extremely small) split, not a bug.

## Installation

Create and activate a virtual environment, then install dependencies.

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Model checkpoints are **not included** in this repository — a trained checkpoint is roughly 169 MB, and the project is designed to be reproduced by training locally rather than by distributing large binary files.

**1. Train the model** (creates `best_transformer_model.pth` locally):
```bash
python train.py
```

**2. Run inference** (loads `best_transformer_model.pth`, translates a hardcoded example sentence):
```bash
python inference.py
```

**3. Run attention analysis** (loads `best_transformer_model.pth`, visualizes encoder attention):
```bash
python attention_analysis.py
```

`inference.py` and `attention_analysis.py` both require `best_transformer_model.pth` to exist, so `train.py` must be run first.

## Training

Configuration exactly as defined in `train.py`:

| Setting | Value |
|---|---|
| Epochs | 20 |
| Batch size | 2 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss | Cross-entropy (padding index ignored) |
| Model dimension (`d_model`) | 512 |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Attention heads | 8 |
| Feed-forward dimension (`d_ff`) | 2048 |
| Dropout | 0.1 |

The best checkpoint (by validation loss) is saved to `best_transformer_model.pth` during training.

## Inference

`inference.py` builds the vocabulary from `data/train.txt`, loads `best_transformer_model.pth`, and greedy-decodes a hardcoded example sentence (`"i love ai"`). In a run against a freshly trained checkpoint, the model produced the single token `"comment"` before predicting `<EOS>` — a translation that is not linguistically correct (the expected output would be closer to "j aime ia"). This is an expected result for a model trained on 8 sentences for a small number of epochs, not a claim of translation quality — it illustrates that the pipeline runs end-to-end, nothing more.

## Attention Analysis

`attention_analysis.py` loads `best_transformer_model.pth` and inspects the **encoder side only**:

- Encoder self-attention weights per layer/head (`model.encoder.layers[i].self_attention_block.attention_weights`), visualized as a heatmap grid across all 8 attention heads for the encoder's first layer
- Per-layer encoder hidden states (`model.encoder.hidden_states`)
- Layer-to-layer cosine similarity of a chosen token's hidden state, as a rough proxy for how much its representation changes as it passes through the encoder stack

Decoder self-attention and cross-attention are **not** visualized — the decoder's attention classes (in `decoder.py`) don't cache their attention weights the way the encoder's `MultiHeadAttention` does, so there is currently no equivalent hook for inspecting them.

## Results / Observations

From a full training run on the current code and dataset:

- Training loss decreases substantially and consistently across all 20 epochs.
- Validation loss reaches its best value around **epoch 7**, then rises for the remainder of training.
- This is a clear sign of overfitting on the 8-example training set — expected given the dataset size, not a defect in the model or training loop.
- Inference runs successfully end-to-end, but translation quality is poor (see Inference above).
- Attention analysis runs successfully and produces the described visualizations without errors.

These are observations from a single tiny experiment, not benchmark results.

## Limitations

- Extremely small dataset (10 sentence pairs total).
- A toy translation task, not a realistic MT benchmark.
- An 8/1/1 train/validation/test split, chosen for illustration rather than statistical validity.
- Validation/test sentences contain out-of-vocabulary words relative to the training-only vocabulary.
- No statistically meaningful evaluation of translation quality is performed or claimed.
- No production-readiness or performance claims are made.
- No decoder self-attention or cross-attention visualization is implemented — only encoder self-attention.
- Padding-mask handling is not currently wired into the encoder's self-attention path (`EncoderBlock.forward` takes no mask argument), so padded positions are not masked out during encoder attention. This is a known architectural gap in the current implementation, not something this project currently addresses.

## Reproducibility

- Trained model checkpoints are intentionally **not** committed to this repository.
- Running `python train.py` regenerates `best_transformer_model.pth` locally in a few seconds, given the dataset size.
- All Python dependencies are listed in `requirements.txt`.
- The full dataset (`data/train.txt`, `data/valid.txt`, `data/test.txt`) is included in the repository.

## Third-Party Attribution

This project is built using:

- [PyTorch](https://pytorch.org/) — model implementation, training, and tensor operations
- [Matplotlib](https://matplotlib.org/) — attention heatmap visualization

Both are used as installed third-party libraries via `pip`; no third-party source code is vendored or copied into this repository. Each library remains governed by its own license.

The Transformer architecture implemented here is based on the original paper:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS).

The architecture itself is not an original invention of this project — this repository is an from-scratch educational reimplementation of that published architecture.

## Learning / Research Context

This project was built to understand Transformer internals by implementing them directly, rather than by calling a pretrained model or a high-level library implementation. It does not present novel research results, and no claims beyond what is described in this README (correct end-to-end implementation of the core architecture, on a small illustrative dataset) should be inferred from this repository.

## License

The source code in this repository is released under the [MIT License](LICENSE). This license applies to this project's own code and does not extend to third-party libraries (PyTorch, Matplotlib), which remain governed by their own respective licenses.
