# AGENTS.md — GPT-Based Appearance Prediction for Deep OC-SORT (MOTS)

## Overview

This document describes the plan to replace the Exponential Moving Average (EMA) embedding
update in Deep OC-SORT with a GPT-based (decoder-only transformer) model that predicts the
next appearance embedding from the full history of a track's appearance embeddings.

The system operates in an **MOTS (Multi-Object Tracking and Segmentation)** setting: detections
include segmentation masks, and object crops are **masked** (background set to black pixels)
before being fed to the image encoder. This reduces background noise and helps the model focus
on the object's appearance.

### Current Design (EMA + ReID)

Each track (`KalmanBoxTracker`) stores a single embedding `self.emb` updated via EMA:

```
self.emb = alpha * self.emb + (1 - alpha) * new_emb
```

Embeddings are extracted using a FastReID network (trained for person re-identification).
At association time, `self.emb` is compared against detection embeddings using cosine similarity
to form the appearance cost matrix.

**Problem:** ReID networks are trained to **discriminate between different objects**, not to
extract visual features that capture **temporal appearance changes**. The embedding space is
optimized for identity separation, not for representing how an object's appearance evolves
over time (pose changes, lighting shifts, partial occlusions, etc.).

### Proposed Design (DINOv3 + GPT Embedding Predictor)

We replace the ReID encoder with **DINOv3-ViT-B**, a self-supervised Vision Transformer that
produces general-purpose visual features better suited for capturing temporal appearance dynamics.

Instead of a single EMA embedding, each track stores the **full history** of appearance
embeddings via an `AppearanceTracker` object. At association time, a GPT model takes the
sequence `[e_1, e_2, ..., e_t]` and predicts `e_{t+1}`, the expected next appearance. This
predicted embedding is then used in the cost matrix, replacing the EMA embedding.

The GPT uses **next-token prediction**: each "token" is a continuous appearance embedding
vector (DINOv3 CLS token, 768-dim). The model is trained in two stages on ground-truth
tracks from MOTS training data.

**Why DINOv3 over ReID:**
- Self-supervised features that work out-of-the-box for distance-based matching
- +6.7 J&F-Mean improvement on video tracking benchmarks (over DINOv2)
- Gram anchoring ensures stable, high-quality dense features across frames
- RoPE positional embeddings generalize well to variable-sized object crops
- 768-dim CLS token provides a compact, semantically rich representation

---

## Image Encoder: DINOv3-ViT-B

**Model:** `facebook/dinov3-vitb16-pretrain` (or equivalent checkpoint)

- **Architecture:** ViT-B/16 (86M params, patch size 16)
- **Output:** CLS token — single 768-dim vector per image crop
- **Pre-training:** Self-supervised on 1.7B images (LVD-1689M dataset)

**Crop preprocessing pipeline:**
1. Crop the bounding box region from the frame
2. Apply the segmentation mask: set background pixels to black (0, 0, 0)
3. Resize the masked crop to 224x224 (ViT input size)
4. Normalize with DINOv3 standard normalization (ImageNet mean/std)
5. Feed to DINOv3-ViT-B → extract CLS token (768-dim)
6. L2-normalize the embedding

This replaces the `EmbeddingComputer` class that currently uses FastReID. The grid patch
mode is removed since DINOv3's patch-based architecture inherently captures spatial structure.

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

**Shared models:**
The `AppearanceGPT` model and `DINOv3Encoder` are shared across all `AppearanceTracker`
instances via class variables, set once in `OCSort.__init__()`.

**Batch prediction:**
For efficiency, `AppearanceTracker.batch_predict(trackers)` processes all tracks in a single
forward pass, rather than calling `predict()` one-by-one.

**Fallback:**
When the GPT model is not loaded, `predict()` falls back to returning the EMA embedding,
preserving the original Deep OC-SORT behavior.

### GPT Embedding Predictor (`AppearanceGPT`)

Standard decoder-only transformer architecture:

```
Input: [e_1, e_2, ..., e_t]  — sequence of L2-normalized DINOv3 CLS token embeddings
  |
  v
Input Projection: Linear(768, d_model)
  |
  v
N x Transformer Decoder Block:
    - Causal (masked) Multi-Head Self-Attention + Relative Position Bias
    - Layer Norm (pre-norm)
    - Feed-Forward Network (MLP with GELU)
    - Layer Norm (pre-norm)
    - Residual connections
  |
  v
Output Projection: Linear(d_model, 768)
  |
  v
L2 Normalize
  |
  v
Output: [pred_e_2, pred_e_3, ..., pred_e_{t+1}]
```

**Hyperparameters:**
- `emb_dim`: 768 (DINOv3-ViT-B CLS token dimension)
- `d_model`: 256
- `n_heads`: 4
- `n_layers`: 4
- `d_ff`: 1024 (feed-forward hidden dim)
- `max_seq_len`: 256
- `dropout`: 0.1

**Key Design Decisions:**
1. The model is small (~2M params) for real-time tracking efficiency.
2. Pre-norm transformer (more stable training).
3. Causal masking ensures each position can only attend to previous positions.
4. **Relative position bias** (no absolute positional embeddings): Each attention head
   learns a scalar bias indexed by `d = query_pos - key_pos`, the distance between query
   and key positions. This means the model encodes "how far back is this observation from
   the current prediction point" rather than "what is its absolute index in the sequence."
   This makes the representation translation-invariant in time.

---

## Training

Training proceeds in two stages:

### Stage 1: Pre-train GPT (ViT frozen)

**Goal:** Teach the GPT to predict temporal appearance dynamics in DINOv3's feature space.

**Objective:** Next-token prediction on appearance embedding sequences.

**Loss:** Negative cosine similarity loss between predicted and actual embeddings:
```
loss = 1 - cosine_similarity(predicted_emb, actual_emb)
```
This is averaged across all valid positions in the sequence.

**Pipeline:**
1. Load KITTI MOTS ground-truth annotations (with segmentation masks and track IDs).
2. For each GT detection: crop bounding box, apply mask (black out background), resize to
   224x224, feed to **frozen** DINOv3-ViT-B → extract 768-dim CLS token.
3. Group embeddings by track ID to form sequences ordered by frame number.
4. Each track of length T yields T-1 training pairs (input prefix → next embedding).

**Training details:**
- Teacher forcing (standard autoregressive training)
- Batch sequences with padding + attention masks
- AdamW optimizer, learning rate 1e-4 with cosine annealing and warmup
- Weight decay: 0.01
- Gradient clipping: max_norm 1.0
- Train for ~50 epochs
- DINOv3-ViT-B weights are **frozen** (no gradients)

### Stage 2: Joint Fine-Tuning (ViT + GPT)

**Goal:** Fine-tune both DINOv3-ViT-B and AppearanceGPT jointly so that the encoding space
becomes more temporally predictable. The pre-trained ViT produces good general visual features
but is not trained to capture temporal appearance dynamics. Joint fine-tuning pushes the ViT
to produce encodings where temporal progression is smooth and predictable by the GPT.

**Models:** DINOv3-ViT-B (unfrozen) + AppearanceGPT (unfrozen)

**Combined loss:** `L = L_pred + λ * L_contrastive`

**Prediction loss** (`L_pred`): Same as Stage 1 — negative cosine similarity between GPT
predictions and actual next embeddings:
```
L_pred = 1 - cosine_similarity(GPT(history), ViT(actual_next_crop))
```
Backpropagates through both GPT and ViT, allowing the encoder to adapt its representations
to be more temporally predictable.

**Contrastive loss** (`L_contrastive`): Triplet or InfoNCE loss that prevents ViT
representation collapse during joint fine-tuning:
- **Anchor:** `ViT(crop_t)` — current frame encoding
- **Positive:** `ViT(crop_{t+1})` — same object, consecutive frame
- **Negatives:**
  - `ViT(crop_{t+k}), k ≠ 1` — same track, non-consecutive (hard negative)
  - `ViT(crop from different track)` — different object

Without the contrastive loss, the ViT could collapse all embeddings to be identical (trivially
making predictions perfect). The contrastive term ensures the ViT maintains discriminative
power while becoming more temporally smooth.

**Data Sampling:**

For each training track with history `[e_1, ..., e_t, e_{t+1}]`, we construct:

| Pair type | Anchor | Other | Source |
|-----------|--------|-------|--------|
| Positive | `ViT(crop_t)` | `ViT(crop_{t+1})` | Same track, consecutive frames |
| Hard negative | `ViT(crop_t)` | `ViT(crop_{t+k}), k ≠ 1` | Same track, non-consecutive |
| Negative | `ViT(crop_t)` | `ViT(crop from different track)` | Different track |

**Training details:**
- Both ViT and GPT update simultaneously (no alternating schedule)
- Lower learning rate for ViT (e.g., 1e-5) vs GPT (e.g., 1e-4)
- ViT is initialized from original DINOv3 pre-trained weights (frozen during Stage 1)
- GPT is initialized from Stage 1 best checkpoint

### Training Data

**Dataset:** KITTI MOTS

KITTI MOTS provides:
- Video sequences with bounding boxes
- Instance segmentation masks for each detection
- Track IDs for temporal association
- Object categories: Car, Pedestrian

**Preprocessing:**
1. For each GT annotation: extract bounding box crop from frame image
2. Apply segmentation mask: pixels outside the mask set to black (0, 0, 0)
3. Resize masked crop to 224x224
4. Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

### Inference Integration

During tracking (online), each `KalmanBoxTracker` owns an `AppearanceTracker`.
DINOv3-ViT-B **fully replaces** FastReID as the image encoder.

**Per-frame pipeline:**
1. **Encode detections:** For each detection (with mask from detector), crop + mask + resize →
   DINOv3-ViT-B → 768-dim CLS token → L2-normalize. This produces `det_embs`.
2. **Predict track embeddings:** `AppearanceTracker.batch_predict(trackers)` runs the GPT on
   all tracks' histories in a single batched forward pass → `trk_embs`.
3. **Association:** Cosine similarity between `det_embs` and `trk_embs` forms the appearance
   cost matrix. Combined with IoU and motion costs via Hungarian algorithm.
4. **Update matched tracks:** `tracker.update(det_emb)` appends the detection's ViT encoding
   to the track's history.
5. **New tracks:** `AppearanceTracker(det_emb)` created with first detection's encoding.

**Fallback:** When the GPT model is not loaded (`--gpt_off`), the system falls back to EMA
embedding updates using DINOv3 encodings (or optionally ReID if `--use_reid` is set).

---

## Files

### `trackers/integrated_ocsort_embedding/dinov3_encoder.py` (NEW)
- `DINOv3Encoder`: Wrapper around DINOv3-ViT-B for extracting CLS token embeddings
  - `__init__(model_name, device)`: Load pre-trained DINOv3-ViT-B
  - `encode(images, masks)`: Batch encode masked crops → (N, 768) L2-normalized embeddings
  - `preprocess(image, bbox, mask)`: Crop, apply mask, resize to 224x224, normalize

### `trackers/integrated_ocsort_embedding/gpt_model.py` (MODIFIED)
- `CausalSelfAttention`: Multi-head causal self-attention with relative position bias
- `TransformerBlock`: Pre-norm transformer decoder block
- `AppearanceGPT(nn.Module)`: The GPT model (updated `emb_dim` default: 768)
- `AppearanceTracker`: Per-track state object (predict/update pattern)
  - Class variable `_model`: shared AppearanceGPT instance
  - Class variable `_encoder`: shared DINOv3Encoder instance
  - `predict()` — predict next embedding from history
  - `update(emb)` — register new observation
  - `get_emb()` — return current predicted embedding
  - `batch_predict(trackers)` — batch-predict for efficiency

### `trackers/integrated_ocsort_embedding/embedding.py` (MODIFIED)
- `EmbeddingComputer`: Updated to support DINOv3 as an alternative encoder
  - Grid mode removed when using DINOv3
  - New `compute_embedding_dinov3(img, bbox, masks, tag)` method
  - Mask application: background pixels set to black before encoding

### `trackers/integrated_ocsort_embedding/ocsort.py` (MODIFIED)
- `KalmanBoxTracker.__init__()`: Creates `self.appearance = AppearanceTracker(emb)`
- `KalmanBoxTracker.update_emb()`: Delegates to `self.appearance.update(emb)`
- `KalmanBoxTracker.predict_emb()`: Delegates to `self.appearance.predict()`
- `KalmanBoxTracker.get_emb()`: Delegates to `self.appearance.get_emb()`
- `OCSort.__init__()`: Loads DINOv3 encoder and GPT model
- `OCSort.update()`: Accepts masks alongside detections; encodes with DINOv3;
  calls `AppearanceTracker.batch_predict()` before association

### `train_gpt.py` (MODIFIED — Stage 1)
- `TrackEmbeddingDataset`: PyTorch dataset for track embedding sequences
- `extract_gt_embeddings_dinov3()`: Extracts DINOv3 embeddings for masked GT crops
- Training loop with validation and checkpointing

### `train_finetune.py` (NEW — Stage 2)
- `FinetuneTrackDataset`: Dataset that provides (anchor, positive, negatives) samples
- `train_finetune()`: Joint fine-tuning loop with prediction + contrastive loss
- Data sampling logic for contrastive pairs (positive, hard negative, negative)
- Checkpoint saving for ViT + GPT

### `main.py` (MODIFIED)
- `--gpt_model_path`: Path to trained GPT model weights
- `--gpt_off`: Flag to disable GPT and fall back to EMA
- `--encoder`: Choice of encoder (`dinov3` or `reid`), default `dinov3`
- `--vit_model_path`: Path to fine-tuned DINOv3 weights (from Stage 2)
- Dataset option updated to include `kitti_mots`

---

## Usage

### Stage 1: Pre-train GPT
```bash
python train_gpt.py --dataset kitti_mots --data_dir data --epochs 50 --encoder dinov3
```

### Stage 2: Joint Fine-tuning
```bash
python train_finetune.py --dataset kitti_mots --data_dir data --epochs 30 \
    --gpt_checkpoint checkpoints/gpt_appearance/best_kitti_mots.pth
```

### Inference (with DINOv3 + GPT)
```bash
python main.py --dataset kitti_mots --encoder dinov3 \
    --vit_model_path checkpoints/finetune/best_vit_kitti_mots.pth \
    --gpt_model_path checkpoints/finetune/best_gpt_kitti_mots.pth
```

### Inference (fallback to EMA with DINOv3)
```bash
python main.py --dataset kitti_mots --encoder dinov3 --gpt_off
```

### Inference (legacy ReID fallback)
```bash
python main.py --dataset kitti_mots --encoder reid --gpt_off
```
