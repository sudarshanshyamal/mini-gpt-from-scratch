# Mini GPT from Scratch

A minimal, fully transparent GPT (Generative Pre-trained Transformer) implementation built from scratch in PyTorch for learning purposes. Train a character-level language model on your own tiny dataset, customize weight initialization, and watch a transformer learn — all on a CPU in under a minute.

> **This project is for educational purposes only.** It is not intended for production use. The goal is to make every piece of the GPT architecture visible, editable, and understandable.

---

## Why This Project?

Most GPT tutorials either gloss over the internals or drown you in a production-scale codebase. This project sits in the sweet spot:

- **~200 lines of heavily commented Python** — nothing is hidden behind library abstractions.
- **Trains in seconds on CPU** — no GPU, no cloud, no waiting.
- **Sub-1KB dataset** — small enough to memorize, which is the point. You can watch the model go from random garbage to coherent text and understand *why*.
- **Custom weight initialization** — swap between Normal, Xavier, Kaiming, or your own scheme and observe how training dynamics change.
- **Full weight inspection** — prints weight statistics before and after training so you can see exactly how parameters evolved.

---

## What You'll Learn

| Concept | Where in the code |
|---|---|
| Character-level tokenization | `encode()` / `decode()` and the vocab dictionaries |
| Token and position embeddings | `MiniGPT.__init__()` — two `nn.Embedding` layers |
| Scaled dot-product self-attention | `SelfAttentionHead.forward()` |
| Causal (autoregressive) masking | The `tril` buffer in `SelfAttentionHead` |
| Multi-head attention | `MultiHeadAttention` — parallel heads, concat, project |
| Feed-forward network with GELU | `FeedForward` — expand → activate → compress |
| Residual connections + LayerNorm | `TransformerBlock.forward()` |
| Weight initialization strategies | `MiniGPT._init_weights()` |
| Cross-entropy loss and backprop | `MiniGPT.forward()` — loss computation |
| Autoregressive text generation | `MiniGPT.generate()` — temperature-controlled sampling |

---

## Requirements

- **Python 3.8+**
- **PyTorch** (CPU build is sufficient)

```bash
pip install torch
```

That's it. No other dependencies.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/sudarshanshyamal/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch

# Run training + generation
python mini_gpt.py
```

You'll see output like this:

```
Dataset size: 498 bytes (498 characters)
Vocabulary (31 tokens): .BCRSTWabcdefghiklmnoprstuvwy

Model created with 105,759 parameters

── Generation BEFORE Training (expect random garbage) ──
av vwdahrtCp.TWlwn.gwtBtT...

── Training ──
  Step     0/2000  |  Loss: 3.4503
  Step   400/2000  |  Loss: 1.2524
  Step  1000/2000  |  Loss: 0.3129
  Step  1999/2000  |  Loss: 0.1494

── Generation AFTER Training ──
  Temperature = 0.5:
  Water flows from the mountains to the sea.
  Birds fly south in winter and return in spring...
```

---

## Project Structure

```
mini-gpt-from-scratch/
├── mini_gpt.py      # The entire model, training loop, and generation
├── README.md
└── mini_gpt_model.pt   # (auto-generated) Saved model checkpoint
```

Everything lives in a single file by design. When learning, having all the pieces in one place beats navigating a modular codebase.

---

## Architecture at a Glance

```
Input text: "The sun"
     │
     ▼
┌──────────────────────┐
│  Character Tokenizer  │   "T"→7, "h"→16, "e"→13, " "→1, ...
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Token Embedding      │   token ID → 64-dim vector (learnable)
│  + Position Embedding │   position  → 64-dim vector (learnable)
└──────────┬───────────┘
           ▼
┌──────────────────────────────────┐
│  Transformer Block ×2            │
│  ┌────────────────────────────┐  │
│  │ Multi-Head Self-Attention  │  │  4 heads, each 16-dim
│  │ (Q·Kᵀ/√d → mask → softmax │  │
│  │  → weighted sum of V)      │  │
│  └────────────┬───────────────┘  │
│  ┌────────────▼───────────────┐  │
│  │ Feed-Forward Network       │  │  64 → 256 → 64 with GELU
│  └────────────────────────────┘  │
│  + Residual connections          │
│  + Layer normalization           │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────┐
│  Output Head          │   64-dim → 31 logits (one per character)
│  → Softmax            │   logits  → probabilities
│  → Sample             │   pick next character
└──────────────────────┘
```

---

## Customizing Weight Initialization

The `_init_weights()` method is where you control how every weight matrix starts before training. Open `mini_gpt.py` and find this section:

```python
def _init_weights(self):
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear):
            # ── OPTION A: Normal distribution (GPT-2 style, active) ──
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # ── OPTION B: Xavier (good for tanh/sigmoid) ──
            # nn.init.xavier_uniform_(module.weight)

            # ── OPTION C: Kaiming (good for ReLU/GELU) ──
            # nn.init.kaiming_normal_(module.weight, mode='fan_in')

            # ── OPTION D: Your own custom values ──
            # nn.init.constant_(module.weight, 0.01)
            # nn.init.uniform_(module.weight, -0.05, 0.05)
```

Comment out Option A and uncomment another to try it. The weight inspection printout before and after training will show you the impact.

---

## Experiments to Try

Here are structured experiments that build understanding progressively:

### 1. Weight Initialization Comparison

Run with each init strategy and compare the loss at step 200:

| Init Strategy | Expected Step-200 Loss | Why |
|---|---|---|
| `normal_(0, 0.02)` | ~1.9 | Small start, stable but cautious |
| `xavier_uniform_` | ~1.7 | Scaled to layer size, good balance |
| `kaiming_normal_` | ~1.6 | Tuned for GELU, fastest convergence |
| `constant_(0.01)` | ~2.5+ | All weights same = broken symmetry issue |
| `normal_(0, 0.5)` | ~3.0+ or NaN | Too large = exploding gradients |

### 2. Model Size

Change hyperparameters and observe:

```python
# Tiny (faster, less capacity)
EMBED_DIM = 32, NUM_HEADS = 2, NUM_LAYERS = 1

# Larger (slower, can memorize more)
EMBED_DIM = 128, NUM_HEADS = 8, NUM_LAYERS = 4
```

### 3. Dataset

Replace the `DATASET` string with your own text:
- Song titles, city names, Python code, chemical formulas — anything.
- Keep it under ~1KB for fast training.
- Observe how the model captures patterns (rhyming, syntax, structure).

### 4. Temperature During Generation

```python
temperature = 0.1   # Nearly deterministic, repetitive
temperature = 0.8   # Balanced (default)
temperature = 1.5   # Creative but potentially nonsensical
```

### 5. Context Window

```python
BLOCK_SIZE = 8    # Can only see 8 characters back — short memory
BLOCK_SIZE = 64   # Sees 64 characters — captures longer patterns
```

---

## Hyperparameter Reference

| Parameter | Default | Description |
|---|---|---|
| `BLOCK_SIZE` | 32 | Context window (how many past characters the model sees) |
| `EMBED_DIM` | 64 | Dimensionality of token/position embeddings |
| `NUM_HEADS` | 4 | Number of parallel attention heads |
| `NUM_LAYERS` | 2 | Number of stacked transformer blocks |
| `DROPOUT` | 0.1 | Dropout probability for regularization |
| `LEARNING_RATE` | 3e-4 | Adam optimizer learning rate |
| `MAX_ITERS` | 2000 | Total training steps |
| `BATCH_SIZE` | 16 | Sequences per training batch |
| `DEVICE` | `cpu` | Set to `cuda` if you have a GPU |

**Constraint:** `EMBED_DIM` must be evenly divisible by `NUM_HEADS`.

---

## How It Compares to Real GPTs

| Aspect | This Project | GPT-2 Small | GPT-3 |
|---|---|---|---|
| Parameters | ~106K | 117M | 175B |
| Vocabulary | 31 (characters) | 50,257 (BPE) | 50,257 (BPE) |
| Context window | 32 | 1,024 | 2,048 |
| Layers | 2 | 12 | 96 |
| Embedding dim | 64 | 768 | 12,288 |
| Training data | ~500 bytes | ~40 GB | ~570 GB |
| Training time | ~10 seconds | Days | Months |

The architecture is *identical* — the same attention mechanism, the same residual connections, the same feed-forward structure. Only the scale differs.

---

## Key Concepts Glossary

- **Token**: The smallest unit the model works with. Here, a single character. In production GPTs, a subword chunk (e.g., "pre", "train", "ing").
- **Embedding**: A learnable vector representation of a token or position. Converts discrete IDs into continuous vectors the model can compute with.
- **Self-Attention**: The mechanism that lets each token "look at" other tokens in the sequence and decide how much information to gather from each.
- **Causal Mask**: Prevents tokens from attending to future positions. This is what makes the model autoregressive — it can only predict the next token based on past context.
- **Residual Connection**: Adding the input back to the output of a sublayer (`x + sublayer(x)`). Helps gradients flow during training and stabilizes deep networks.
- **Layer Normalization**: Normalizes activations within each layer to stabilize training.
- **Temperature**: A scaling factor applied to logits before softmax during generation. Lower = more deterministic, higher = more random.

---

## Further Reading

Resources that pair well with this project:

- [Andrej Karpathy — "Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the video that inspired much of this approach
- [Andrej Karpathy — "Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) — the full series from basics to GPT
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — visual explanations of attention
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original transformer paper
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Language Models are Unsupervised Multitask Learners

---

## License

This project is released under the [MIT License](LICENSE). Use it, modify it, learn from it.

---

## Acknowledgments

Built as a hands-on learning exercise inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his "Zero to Hero" lecture series. The goal was to create the smallest possible GPT that still teaches the real architecture.

---

*If this helped you understand transformers, consider starring the repo — it helps others find it.*
