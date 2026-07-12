"""
=============================================================================
  Mini GPT from Scratch — A Learning-Focused Implementation
=============================================================================
  
  This script builds a tiny GPT (Generative Pre-trained Transformer) that you
  can train on your own small text dataset (< 1KB) on a Windows CPU machine.
  
  What you'll learn:
    1. How tokenization works (character-level here)
    2. How the Transformer architecture is built block by block
    3. How to define and customize your own weight initialization
    4. How the training loop works
    5. How text generation (inference) works
  
  Requirements:
    pip install torch
  
  Usage:
    python mini_gpt.py
  
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: YOUR DATASET  (< 1KB — just edit this string!)
# ──────────────────────────────────────────────────────────────────────────────
# The model will learn patterns in this text and try to generate similar text.
# Feel free to replace this with anything you like.

DATASET = """
The sun rises in the east and sets in the west.
Water flows from the mountains to the sea.
Birds fly south in winter and return in spring.
Trees grow tall when they receive enough sunlight.
The moon orbits the earth and the earth orbits the sun.
Rivers carve valleys through the landscape over time.
Clouds form when water vapor rises and cools in the sky.
Seeds need water and warmth to sprout and grow.
The wind carries seeds to new places far away.
Stars shine brightly in the clear night sky.
"""

print(f"Dataset size: {len(DATASET)} bytes ({len(DATASET)} characters)")
print(f"Preview: {DATASET[:80]}...\n")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: CHARACTER-LEVEL TOKENIZER
# ──────────────────────────────────────────────────────────────────────────────
# Real GPTs use subword tokenizers (BPE). For learning, character-level is
# simpler: each unique character gets an integer ID.

chars = sorted(list(set(DATASET)))       # all unique characters
vocab_size = len(chars)                   # how many unique tokens we have
print(f"Vocabulary ({vocab_size} tokens): {''.join(chars)}\n")

# Mapping dictionaries: character <-> integer
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode/decode functions
def encode(text):
    """Convert a string into a list of integer token IDs."""
    return [char_to_idx[ch] for ch in text]

def decode(token_ids):
    """Convert a list of integer token IDs back into a string."""
    return ''.join([idx_to_char[i] for i in token_ids])

# Encode the entire dataset into a tensor
data = torch.tensor(encode(DATASET), dtype=torch.long)
print(f"Encoded dataset shape: {data.shape}")
print(f"Sample encoding: '{DATASET[1:11]}' -> {encode(DATASET[1:11])}\n")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: HYPERPARAMETERS — All tunable, all yours to experiment with!
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE    = 32      # Context window: how many past chars the model sees
EMBED_DIM     = 64      # Size of each token's embedding vector
NUM_HEADS     = 4       # Number of attention heads (must divide EMBED_DIM)
NUM_LAYERS    = 2       # Number of transformer blocks stacked
DROPOUT       = 0.1     # Dropout rate for regularization
LEARNING_RATE = 3e-4    # Adam optimizer learning rate
MAX_ITERS     = 2000    # Total training iterations
EVAL_INTERVAL = 200     # Print loss every N iterations
BATCH_SIZE    = 16      # Number of training sequences per batch
DEVICE        = 'cpu'   # Use 'cuda' if you have a GPU; 'cpu' works fine here

print("─── Hyperparameters ───")
print(f"  Block size (context length) : {BLOCK_SIZE}")
print(f"  Embedding dimension         : {EMBED_DIM}")
print(f"  Attention heads             : {NUM_HEADS}")
print(f"  Transformer layers          : {NUM_LAYERS}")
print(f"  Device                      : {DEVICE}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: DATA LOADER — Creates random training batches
# ──────────────────────────────────────────────────────────────────────────────

def get_batch():
    """
    Grab a random batch of input-target pairs from the dataset.
    
    For each sequence:
      - Input  (x): characters at positions [i, i+1, ..., i+BLOCK_SIZE-1]
      - Target (y): characters at positions [i+1, i+2, ..., i+BLOCK_SIZE]
    
    The model learns to predict the NEXT character at every position.
    """
    # Pick random starting indices
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i   : i + BLOCK_SIZE]     for i in ix])
    y = torch.stack([data[i+1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: TRANSFORMER BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────

class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention.
    
    This is the core mechanism: each token "looks at" all previous tokens
    and computes a weighted combination of their values.
    
    Q (Query)  = "What am I looking for?"
    K (Key)    = "What do I contain?"
    V (Value)  = "What information do I provide?"
    """
    def __init__(self, head_dim):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.key   = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        
        # Causal mask: prevents attending to future tokens
        # (a GPT can only look at past + present, not the future)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
    
    def forward(self, x):
        B, T, C = x.shape       # Batch, Time (seq length), Channels (embed dim)
        
        q = self.query(x)       # (B, T, head_dim)
        k = self.key(x)         # (B, T, head_dim)
        v = self.value(x)       # (B, T, head_dim)
        
        # Attention scores: how much should each token attend to each other?
        # Scale by sqrt(head_dim) for stable gradients
        scale = math.sqrt(k.shape[-1])
        attn = (q @ k.transpose(-2, -1)) / scale    # (B, T, T)
        
        # Apply causal mask: set future positions to -infinity so softmax → 0
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)               # (B, T, T)
        attn = self.dropout(attn)
        
        # Weighted combination of values
        out = attn @ v                                # (B, T, head_dim)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel, then concatenated.
    Each head can learn to focus on different types of relationships.
    """
    def __init__(self):
        super().__init__()
        head_dim = EMBED_DIM // NUM_HEADS
        self.heads = nn.ModuleList([SelfAttentionHead(head_dim) for _ in range(NUM_HEADS)])
        self.projection = nn.Linear(EMBED_DIM, EMBED_DIM)  # mix head outputs
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        # Run all heads, concatenate, then project
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """
    A simple two-layer MLP applied to each token independently.
    This is where the model "thinks" after gathering attention info.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),    # expand
            nn.GELU(),                                # activation function
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),    # compress back
            nn.Dropout(DROPOUT),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block = Attention + FeedForward, each with
    residual connections and layer normalization.
    """
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention()
        self.feedforward = FeedForward()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
    
    def forward(self, x):
        # Pre-norm architecture (as used in GPT-2/3)
        x = x + self.attention(self.ln1(x))     # residual + attention
        x = x + self.feedforward(self.ln2(x))   # residual + FFN
        return x


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: THE FULL GPT MODEL
# ──────────────────────────────────────────────────────────────────────────────

class MiniGPT(nn.Module):
    """
    The complete GPT language model:
      1. Token embedding: converts token IDs → vectors
      2. Position embedding: encodes where each token sits in the sequence
      3. Transformer blocks: the attention + FFN layers
      4. Output head: maps final vectors → probability over vocabulary
    """
    def __init__(self):
        super().__init__()
        
        # Embedding layers
        self.token_embedding    = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        
        # Final layer norm + output projection to vocabulary
        self.final_ln = nn.LayerNorm(EMBED_DIM)
        self.output_head = nn.Linear(EMBED_DIM, vocab_size)
        
        # ── CUSTOM WEIGHT INITIALIZATION ──
        # This is where YOU define how weights start!
        # Different initializations lead to different training dynamics.
        self._init_weights()
        
        # Count and display parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {n_params:,} parameters")
        print(f"  Token embedding : {self.token_embedding.weight.shape}")
        print(f"  Position embed  : {self.position_embedding.weight.shape}")
        print(f"  Transformer     : {NUM_LAYERS} blocks × (Attn + FFN)")
        print(f"  Output head     : {self.output_head.weight.shape}")
        print()
    
    def _init_weights(self):
        """
        ╔══════════════════════════════════════════════════════════════╗
        ║  CUSTOMIZE YOUR WEIGHT INITIALIZATION HERE!                 ║
        ║                                                             ║
        ║  Try changing these and observe how training changes:       ║
        ║  • Normal vs Uniform vs Xavier vs Kaiming                   ║
        ║  • Different std values (0.01 vs 0.02 vs 0.1)              ║
        ║  • Zero init for biases vs small random values              ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        print("─── Initializing weights (CUSTOMIZE THIS!) ───")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # ─── OPTION A: Normal distribution (GPT-2 style) ───
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
                # ─── OPTION B: Xavier (good for tanh/sigmoid) ───
                # nn.init.xavier_uniform_(module.weight)
                
                # ─── OPTION C: Kaiming (good for ReLU/GELU) ───
                # nn.init.kaiming_normal_(module.weight, mode='fan_in')
                
                # ─── OPTION D: Your own custom values! ───
                # nn.init.constant_(module.weight, 0.01)  # all same value
                # nn.init.uniform_(module.weight, -0.05, 0.05)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)    # scale = 1
                nn.init.zeros_(module.bias)     # shift = 0
        
        print("  Linear layers  : Normal(mean=0, std=0.02)")
        print("  Embeddings     : Normal(mean=0, std=0.02)")
        print("  LayerNorm      : weight=1, bias=0")
        print("  (Edit _init_weights() to try other strategies!)")
        print()
    
    def forward(self, idx, targets=None):
        """
        Forward pass: tokens in → logits out (and loss if targets given).
        
        idx:     (B, T) tensor of token indices
        targets: (B, T) tensor of next-token targets (optional)
        """
        B, T = idx.shape
        
        # 1. Look up token embeddings + add positional embeddings
        tok_emb = self.token_embedding(idx)                           # (B, T, EMBED_DIM)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))  # (T, EMBED_DIM)
        x = tok_emb + pos_emb                                        # (B, T, EMBED_DIM)
        
        # 2. Pass through transformer blocks
        x = self.blocks(x)                                            # (B, T, EMBED_DIM)
        
        # 3. Final layer norm + project to vocabulary size
        x = self.final_ln(x)                                          # (B, T, EMBED_DIM)
        logits = self.output_head(x)                                  # (B, T, vocab_size)
        
        # 4. Compute loss if training
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: flatten batch and time dimensions
            logits_flat  = logits.view(B * T, vocab_size)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """
        Generate text auto-regressively, one token at a time.
        
        temperature: controls randomness
          - Low  (0.1): very predictable, repetitive
          - High (1.5): very random, creative but may be nonsensical
          - 1.0: balanced
        """
        for _ in range(max_new_tokens):
            # Crop to the last BLOCK_SIZE tokens (model's context window)
            idx_crop = idx[:, -BLOCK_SIZE:]
            
            # Get predictions
            logits, _ = self(idx_crop)
            
            # Focus on the last time step (the prediction for the NEXT token)
            logits = logits[:, -1, :] / temperature    # (B, vocab_size)
            
            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)
            
            # Append to the sequence
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: INSPECT WEIGHTS (see what you initialized!)
# ──────────────────────────────────────────────────────────────────────────────

def inspect_weights(model, label=""):
    """Print statistics about the model's current weight values."""
    print(f"─── Weight Inspection {label} ───")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            print(f"  {name:45s} | shape {str(list(param.shape)):20s} | "
                  f"mean={param.mean():.5f}  std={param.std():.5f}  "
                  f"min={param.min():.5f}  max={param.max():.5f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8: TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 65)
    print("  MINI GPT — Training from Scratch")
    print("=" * 65)
    print()
    
    # Create the model
    model = MiniGPT().to(DEVICE)
    
    # Inspect initial weights
    inspect_weights(model, "(Before Training)")
    
    # Optimizer — Adam is standard for transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # ── Generate BEFORE training (should be garbage) ──
    print("─── Generation BEFORE Training (expect random garbage) ───")
    seed = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = model.generate(seed, max_new_tokens=100)
    print(decode(generated[0].tolist()))
    print()
    
    # ── Training loop ──
    print("─── Training ───")
    for step in range(MAX_ITERS):
        # Get a batch of data
        xb, yb = get_batch()
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Print progress
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            print(f"  Step {step:5d}/{MAX_ITERS}  |  Loss: {loss.item():.4f}")
    
    print()
    
    # Inspect weights after training — compare with before!
    inspect_weights(model, "(After Training)")
    
    # ── Generate AFTER training (should resemble the dataset) ──
    print("─── Generation AFTER Training ───")
    print("(Generating 3 samples with different temperatures)\n")
    
    for temp in [0.5, 0.8, 1.2]:
        print(f"  Temperature = {temp}:")
        seed = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        generated = model.generate(seed, max_new_tokens=150, temperature=temp)
        text = decode(generated[0].tolist())
        # Clean up for display
        print(f"  {text[:150]}")
        print()
    
    # ── Save the model ──
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'hyperparameters': {
            'vocab_size': vocab_size,
            'block_size': BLOCK_SIZE,
            'embed_dim': EMBED_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
        }
    }, 'mini_gpt_model.pt')
    print("Model saved to mini_gpt_model.pt")
    print("Done! Edit the DATASET, hyperparameters, or _init_weights() and run again.")


if __name__ == "__main__":
    train()
