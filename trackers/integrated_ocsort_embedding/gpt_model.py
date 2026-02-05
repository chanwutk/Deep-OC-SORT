"""
GPT-based appearance embedding predictor for multi-object tracking.

Instead of using an Exponential Moving Average (EMA) to represent a track's
appearance, this module uses a decoder-only transformer (GPT) to predict the
next appearance embedding from the full history of a track's embeddings.

Design mirrors KalmanBoxTracker from OC-SORT:
    - AppearanceTracker.predict()  — predict next appearance embedding
    - AppearanceTracker.update()   — register an observed appearance embedding

Architecture:
    Input:  (batch, seq_len, emb_dim)   -- historical ReID embeddings
    Output: (batch, seq_len, emb_dim)   -- predicted next embedding at each position

Training objective: next-token prediction with cosine similarity loss.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: upper-triangular = -inf
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            pad_mask: (batch, seq_len) — True for positions to mask (padding)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply padding mask if provided
        if pad_mask is not None:
            # pad_mask: (B, T) -> (B, 1, 1, T) to broadcast over heads and query positions
            attn = attn.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class TransformerBlock(nn.Module):
    """Pre-norm transformer decoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class AppearanceGPT(nn.Module):
    """
    Decoder-only transformer for next-appearance-embedding prediction.

    Takes a sequence of L2-normalized ReID embeddings and predicts the
    next embedding at each position (autoregressive).

    Args:
        emb_dim:     Dimension of input/output embeddings (512 for non-grid, 1536 for grid)
        d_model:     Transformer hidden dimension
        n_heads:     Number of attention heads
        n_layers:    Number of transformer blocks
        d_ff:        Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout:     Dropout rate
    """

    def __init__(
        self,
        emb_dim: int = 512,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection: emb_dim -> d_model
        self.input_proj = nn.Linear(emb_dim, d_model)

        # Learned positional encoding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout) for _ in range(n_layers)]
        )

        # Final layer norm (pre-norm architecture)
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection: d_model -> emb_dim
        self.output_proj = nn.Linear(d_model, emb_dim)

        self.drop = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        emb_seq: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            emb_seq:  (batch, seq_len, emb_dim) — L2-normalized embedding sequence
            pad_mask: (batch, seq_len) — True for padding positions

        Returns:
            (batch, seq_len, emb_dim) — predicted next embeddings, L2-normalized
        """
        B, T, _ = emb_seq.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Project input embeddings
        h = self.input_proj(emb_seq)  # (B, T, d_model)

        # Add positional encoding
        positions = torch.arange(T, device=emb_seq.device)
        h = self.drop(h + self.pos_emb(positions))

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, pad_mask)

        # Final layer norm and output projection
        h = self.ln_f(h)
        out = self.output_proj(h)  # (B, T, emb_dim)

        # L2 normalize output
        out = F.normalize(out, dim=-1)

        return out

    def predict_next(
        self,
        emb_seq: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict only the next embedding (from the last valid position).

        Args:
            emb_seq:  (batch, seq_len, emb_dim)
            pad_mask: (batch, seq_len) — True for padding positions

        Returns:
            (batch, emb_dim) — predicted next embedding for each sequence
        """
        out = self.forward(emb_seq, pad_mask)  # (B, T, emb_dim)

        if pad_mask is not None:
            # Find last valid position for each sequence
            valid_mask = ~pad_mask  # (B, T)
            last_valid_idx = valid_mask.sum(dim=1) - 1  # (B,)
            last_valid_idx = last_valid_idx.clamp(min=0)
            pred = out[torch.arange(out.shape[0], device=out.device), last_valid_idx]
        else:
            pred = out[:, -1, :]  # (B, emb_dim)

        return pred


# ---------------------------------------------------------------------------
# AppearanceTracker — per-track state, mirrors KalmanBoxTracker
# ---------------------------------------------------------------------------


class AppearanceTracker(object):
    """
    Per-track appearance state tracker using GPT-based next-embedding prediction.

    Designed to mirror KalmanBoxTracker's predict/update lifecycle:
        - predict()  — predict the next appearance embedding (like KF predict)
        - update()   — register a new observed appearance embedding (like KF update)
        - get_emb()  — return the current predicted embedding (like get_state)

    The AppearanceGPT model is shared across all AppearanceTracker instances
    via a class variable, similar to KalmanBoxTracker.count.

    Lifecycle per frame (mirrors KalmanBoxTracker):
        1. tracker.predict()       — run before association
        2. association using get_emb()
        3. tracker.update(emb)     — run after matching with detection embedding
    """

    # ---- Shared state across all instances ----
    _model = None       # AppearanceGPT model (shared)
    _grid_off = True    # Whether grid patches are disabled

    @classmethod
    def set_model(cls, model: Optional[AppearanceGPT], grid_off: bool = True):
        """
        Set the shared GPT model for all AppearanceTracker instances.
        Called once during OCSort initialization.
        """
        cls._model = model
        cls._grid_off = grid_off
        if model is not None:
            model.eval()

    @classmethod
    def load_model(cls, checkpoint_path: str, grid_off: bool = True, device: str = "cuda"):
        """
        Load the shared GPT model from a checkpoint file.
        Called once during OCSort initialization.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        model = AppearanceGPT(
            emb_dim=config["emb_dim"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            max_seq_len=config["max_seq_len"],
            dropout=0.0,  # No dropout at inference
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        cls._model = model
        cls._grid_off = grid_off

    # ---- Per-instance state ----

    def __init__(self, emb=None, max_seq_len: int = 256):
        """
        Initialize the appearance tracker with an initial observation.

        Args:
            emb: Initial appearance embedding.
                 Shape: (emb_dim,) for non-grid, (3, emb_dim) for grid.
            max_seq_len: Maximum history length (oldest entries are dropped).
        """
        self.max_seq_len = max_seq_len

        # Observation history: ordered list of observed embeddings
        self.observations = [emb] if emb is not None else []

        # Current predicted embedding (result of predict())
        self._predicted_emb = emb

        # EMA embedding (fallback when model is not available)
        self._ema_emb = emb

    def predict(self):
        """
        Predict the next appearance embedding from the observation history.

        Analogous to KalmanBoxTracker.predict() which advances the Kalman
        state and returns the predicted bounding box.

        Returns:
            np.ndarray — predicted next appearance embedding, or None if
            no observations exist.
        """
        if len(self.observations) == 0:
            return self._predicted_emb

        if self._model is None:
            # Fallback: use EMA embedding as prediction
            self._predicted_emb = self._ema_emb
            return self._predicted_emb

        # Prepare input sequence
        seq = self.observations[-self.max_seq_len :]

        # Flatten grid embeddings: (3, 512) -> (1536,)
        processed = []
        for emb in seq:
            if emb.ndim == 2:
                processed.append(emb.reshape(-1))
            else:
                processed.append(emb)

        emb_seq = np.stack(processed)  # (T, flat_emb_dim)
        emb_tensor = torch.from_numpy(emb_seq).float().unsqueeze(0)  # (1, T, flat_emb_dim)

        device = next(self._model.parameters()).device
        emb_tensor = emb_tensor.to(device)

        with torch.no_grad():
            pred = self._model.predict_next(emb_tensor)  # (1, flat_emb_dim)

        pred_np = pred.squeeze(0).cpu().numpy()

        # Reshape back to grid format if needed
        if not self._grid_off and self.observations[0].ndim == 2:
            pred_np = pred_np.reshape(self.observations[0].shape)

        self._predicted_emb = pred_np
        return self._predicted_emb

    def update(self, emb, alpha=0.9):
        """
        Register a new appearance embedding observation.

        Analogous to KalmanBoxTracker.update(bbox) which incorporates a
        matched detection into the Kalman state.

        Args:
            emb: Observed appearance embedding from the matched detection.
                 Shape: (emb_dim,) for non-grid, (3, emb_dim) for grid.
            alpha: EMA weight for fallback mode. Higher alpha means more
                   weight on the existing embedding (less change).
        """
        if emb is None:
            return

        self.observations.append(emb)
        if len(self.observations) > self.max_seq_len:
            self.observations = self.observations[-self.max_seq_len :]

        # Update EMA embedding (fallback)
        if self._ema_emb is not None:
            self._ema_emb = alpha * self._ema_emb + (1 - alpha) * emb
            self._ema_emb /= np.linalg.norm(self._ema_emb)
        else:
            self._ema_emb = emb

    def get_emb(self):
        """
        Return the current predicted appearance embedding.

        Analogous to KalmanBoxTracker.get_state() which returns the
        current bounding box estimate.

        Should be called after predict() to get the latest prediction.
        """
        return self._predicted_emb

    # ---- Batch prediction for efficiency ----

    @classmethod
    @torch.no_grad()
    def batch_predict(cls, trackers: list):
        """
        Batch-predict next embeddings for multiple AppearanceTrackers.

        More efficient than calling predict() one-by-one because it
        batches all sequences into a single forward pass.

        Args:
            trackers: List of AppearanceTracker instances.
        """
        if cls._model is None or len(trackers) == 0:
            # Fallback: use EMA for all trackers
            for trk in trackers:
                trk._predicted_emb = trk._ema_emb
            return

        # Collect and flatten sequences
        sequences = []
        for trk in trackers:
            seq = trk.observations[-trk.max_seq_len :]
            processed = []
            for emb in seq:
                if emb.ndim == 2:
                    processed.append(emb.reshape(-1))
                else:
                    processed.append(emb)
            sequences.append(processed)

        # Determine max length and embedding dim
        max_len = max(len(s) for s in sequences)
        emb_dim = sequences[0][0].shape[0]

        # Pad sequences and create mask
        batch_size = len(sequences)
        batch_embs = np.zeros((batch_size, max_len, emb_dim), dtype=np.float32)
        pad_mask = np.ones((batch_size, max_len), dtype=bool)  # True = padding

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            batch_embs[i, :seq_len] = np.stack(seq)
            pad_mask[i, :seq_len] = False

        # Run model
        device = next(cls._model.parameters()).device
        batch_embs_t = torch.from_numpy(batch_embs).to(device)
        pad_mask_t = torch.from_numpy(pad_mask).to(device)

        predicted = cls._model.predict_next(batch_embs_t, pad_mask_t)  # (B, emb_dim)
        predicted = predicted.cpu().numpy()

        # Assign predictions back to each tracker
        for i, trk in enumerate(trackers):
            pred = predicted[i]
            # Reshape to grid format if needed
            if not cls._grid_off and len(trk.observations) > 0 and trk.observations[0].ndim == 2:
                pred = pred.reshape(trk.observations[0].shape)
            trk._predicted_emb = pred
