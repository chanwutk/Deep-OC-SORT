# AGENTS.md — GPT-Based Appearance Prediction for Deep OC-SORT

## Overview

This document describes the plan to replace the Exponential Moving Average (EMA) embedding
update in Deep OC-SORT with a GPT-based (decoder-only transformer) model that predicts the
next appearance embedding from the full history of a track's appearance embeddings.

### Current Design (EMA)

Each track (`KalmanBoxTracker`) stores a single embedding `self.emb` updated via EMA:

```
self.emb = alpha * self.emb + (1 - alpha) * new_emb
```

At association time, `self.emb` is compared against detection embeddings using cosine similarity
to form the appearance cost matrix.

### Proposed Design (GPT Embedding Predictor)

Instead of a single EMA embedding, each track stores the **full history** of appearance
embeddings via an `AppearanceTracker` object. At association time, a GPT model takes the
sequence `[e_1, e_2, ..., e_t]` and predicts `e_{t+1}`, the expected next appearance. This
predicted embedding is then used in the cost matrix, replacing the EMA embedding.

The GPT uses **next-token prediction**: each "token" is a continuous appearance embedding
vector (not a discrete vocabulary token). The model is trained autoregressively on ground-truth
tracks from MOT training data.

---

## Architecture

### AppearanceTracker (per-track, mirrors KalmanBoxTracker)

Each track owns an `AppearanceTracker` instance that follows the same predict/update lifecycle
as `KalmanBoxTracker`:

```
KalmanBoxTracker (bbox)          AppearanceTracker (appearance)
─────────────────────────        ─────────────────────────────
predict()  → predicted bbox      predict()  → predicted embedding
update(bbox) → register obs      update(emb) → register observation
get_state()  → current bbox      get_emb()   → current embedding
```

**Per-frame lifecycle:**
1. `tracker.predict()` — run before association (advances state)
2. Association uses `tracker.get_emb()` for the cost matrix
3. `tracker.update(emb)` — run after matching with detection embedding

**Shared model:**
The `AppearanceGPT` model is shared across all `AppearanceTracker` instances via a class
variable (`AppearanceTracker._model`), analogous to `KalmanBoxTracker.count`. Set once in
`OCSort.__init__()`.

**Batch prediction:**
For efficiency, `AppearanceTracker.batch_predict(trackers)` processes all tracks in a single
forward pass, rather than calling `predict()` one-by-one.

**Fallback:**
When the GPT model is not loaded, `predict()` falls back to returning the EMA embedding,
preserving the original Deep OC-SORT behavior.

### GPT Embedding Predictor (`AppearanceGPT`)

Standard decoder-only transformer architecture:

```
Input: [e_1, e_2, ..., e_t]  — sequence of L2-normalized ReID embeddings
  |
  v
Input Projection: Linear(emb_dim, d_model)
  |
  v
Positional Encoding (learned)
  |
  v
N x Transformer Decoder Block:
    - Causal (masked) Multi-Head Self-Attention
    - Layer Norm (pre-norm)
    - Feed-Forward Network (MLP with GELU)
    - Layer Norm (pre-norm)
    - Residual connections
  |
  v
Output Projection: Linear(d_model, emb_dim)
  |
  v
L2 Normalize
  |
  v
Output: [pred_e_2, pred_e_3, ..., pred_e_{t+1}]
```

**Hyperparameters:**
- `emb_dim`: 512 (non-grid) or 1536 (grid, 3 patches x 512)
- `d_model`: 256
- `n_heads`: 4
- `n_layers`: 4
- `d_ff`: 1024 (feed-forward hidden dim)
- `max_seq_len`: 256
- `dropout`: 0.1

**Key Design Decisions:**
1. Grid mode support: When grid mode is active, each detection has shape `(3, 512)`.
   We flatten to 1536-dim as the token, and reshape back to `(3, 512)` after prediction.
2. The model is small (~2M params) for real-time tracking efficiency.
3. Pre-norm transformer (more stable training).
4. Causal masking ensures each position can only attend to previous positions.

### Training

**Objective:** Next-token prediction on appearance embedding sequences.

**Loss:** Negative cosine similarity loss between predicted and actual embeddings:
```
loss = 1 - cosine_similarity(predicted_emb, actual_emb)
```
This is averaged across all valid positions in the sequence.

**Training Data:**
1. Use MOT ground-truth annotations (COCO format) which provide track IDs per detection.
2. For each video sequence, extract ReID embeddings for all GT bounding boxes.
3. Group embeddings by track ID to form sequences ordered by frame number.
4. Each track of length T yields T-1 training pairs (input prefix -> next embedding).

**Training details:**
- Teacher forcing (standard autoregressive training)
- Batch sequences with padding + attention masks
- Adam optimizer, learning rate 1e-4 with cosine annealing
- Train for ~50 epochs
- Supports MOT17, MOT20, DanceTrack datasets

### Inference Integration

During tracking (online), each `KalmanBoxTracker` owns an `AppearanceTracker`:

1. **Before association:** `AppearanceTracker.batch_predict(trackers)` predicts next embeddings
   for all tracks in a single batched forward pass. Each tracker's `_predicted_emb` is updated.
2. **During association:** `tracker.get_emb()` returns the predicted embedding for the cost
   matrix.
3. **After matching:** `tracker.update(emb)` registers the matched detection's embedding.
4. **New tracks:** `AppearanceTracker(emb)` is created with the first detection's embedding.

---

## Files

### `trackers/integrated_ocsort_embedding/gpt_model.py`
- `CausalSelfAttention`: Multi-head causal self-attention
- `TransformerBlock`: Pre-norm transformer decoder block
- `AppearanceGPT(nn.Module)`: The GPT model
- `AppearanceTracker`: Per-track state object (predict/update pattern)
  - Class variable `_model`: shared AppearanceGPT instance
  - `predict()` — predict next embedding from history
  - `update(emb)` — register new observation
  - `get_emb()` — return current predicted embedding
  - `batch_predict(trackers)` — batch-predict for efficiency

### `trackers/integrated_ocsort_embedding/ocsort.py`
- `KalmanBoxTracker.__init__()`: Creates `self.appearance = AppearanceTracker(emb)`
- `KalmanBoxTracker.update_emb()`: Delegates to `self.appearance.update(emb)`
- `KalmanBoxTracker.predict_emb()`: Delegates to `self.appearance.predict()`
- `KalmanBoxTracker.get_emb()`: Delegates to `self.appearance.get_emb()`
- `OCSort.__init__()`: Sets shared model via `AppearanceTracker.load_model(path)`
- `OCSort.update()`: Calls `AppearanceTracker.batch_predict()` before association

### `train_gpt.py`
- `TrackEmbeddingDataset`: PyTorch dataset for track embedding sequences
- `extract_gt_embeddings()`: Extracts ReID embeddings for GT boxes
- Training loop with validation and checkpointing

### `main.py`
- `--gpt_model_path`: Path to trained GPT model weights
- `--gpt_off`: Flag to disable GPT and fall back to EMA

---

## Usage

### Training
```bash
python train_gpt.py --dataset dance --data_dir data --epochs 50 --grid_off
```

### Inference (with GPT)
```bash
python main.py --dataset dance --gpt_model_path checkpoints/gpt_appearance/best_dance.pth
```

### Inference (fallback to EMA)
```bash
python main.py --dataset dance --gpt_off
```
