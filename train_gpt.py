"""
Training script for the GPT-based appearance embedding predictor.

Extracts ReID embeddings from ground-truth bounding boxes, groups them by track,
and trains a decoder-only transformer to predict the next appearance embedding
from the history of a track's embeddings.

Usage:
    python train_gpt.py --dataset mot17 --data_dir data --epochs 50 --batch_size 32
    python train_gpt.py --dataset dance --data_dir data --epochs 50 --grid_off
"""

import argparse
import json
import math
import os
import pickle
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trackers.integrated_ocsort_embedding.embedding import EmbeddingComputer
from trackers.integrated_ocsort_embedding.gpt_model import AppearanceGPT


# ---------------------------------------------------------------------------
# KITTI MOTS sequence splits
# ---------------------------------------------------------------------------

KITTI_MOTS_TRAIN_SEQS = ["0000", "0001", "0003", "0004", "0005", "0009",
                          "0011", "0012", "0015", "0017", "0019", "0020"]
KITTI_MOTS_VAL_SEQS   = ["0002", "0006", "0007", "0008", "0010",
                          "0013", "0014", "0016", "0018"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrackEmbeddingDataset(Dataset):
    """
    Dataset of track embedding sequences for training the GPT predictor.

    Each sample is a (input_seq, target_seq) pair:
        input_seq:  [e_1, e_2, ..., e_{T-1}]
        target_seq: [e_2, e_3, ..., e_T]

    The model is trained to predict target_seq from input_seq (next-token prediction).
    """

    def __init__(self, track_sequences, max_seq_len=256, min_track_len=2):
        """
        Args:
            track_sequences: List of np.ndarray, each with shape (T, emb_dim) or (T, 3, emb_dim)
            max_seq_len: Maximum sequence length (longer tracks are split into chunks)
            min_track_len: Minimum track length to include
        """
        self.max_seq_len = max_seq_len
        self.samples = []

        for seq in track_sequences:
            if len(seq) < min_track_len:
                continue

            # Flatten grid embeddings: (T, 3, 512) -> (T, 1536)
            if seq.ndim == 3:
                T, G, D = seq.shape
                seq = seq.reshape(T, G * D)

            # Split long sequences into overlapping chunks
            if len(seq) <= max_seq_len:
                self.samples.append(seq)
            else:
                # Sliding window with stride = max_seq_len // 2
                stride = max(max_seq_len // 2, 1)
                for start in range(0, len(seq) - min_track_len + 1, stride):
                    end = min(start + max_seq_len, len(seq))
                    chunk = seq[start:end]
                    if len(chunk) >= min_track_len:
                        self.samples.append(chunk)

        print(f"TrackEmbeddingDataset: {len(self.samples)} samples from "
              f"{len(track_sequences)} tracks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]  # (T, emb_dim)
        T = len(seq)

        # Input: all tokens, Target: shifted by 1
        # The model at position i predicts position i+1
        # input:  [e_1, e_2, ..., e_T]
        # target: [e_2, e_3, ..., e_T]  (shifted, length T-1)
        # We train on positions 0..T-2, predicting positions 1..T-1
        input_seq = torch.from_numpy(seq).float()          # (T, emb_dim)
        target_seq = torch.from_numpy(seq[1:]).float()      # (T-1, emb_dim)

        return input_seq, target_seq


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads to the maximum length in the batch.
    """
    input_seqs, target_seqs = zip(*batch)

    # Get maximum lengths
    max_input_len = max(s.shape[0] for s in input_seqs)
    max_target_len = max(s.shape[0] for s in target_seqs)
    emb_dim = input_seqs[0].shape[1]
    batch_size = len(input_seqs)

    # Pad input sequences
    padded_inputs = torch.zeros(batch_size, max_input_len, emb_dim)
    input_pad_mask = torch.ones(batch_size, max_input_len, dtype=torch.bool)  # True = padding

    # Pad target sequences
    padded_targets = torch.zeros(batch_size, max_target_len, emb_dim)
    target_mask = torch.zeros(batch_size, max_target_len, dtype=torch.bool)  # True = valid

    for i, (inp, tgt) in enumerate(zip(input_seqs, target_seqs)):
        inp_len = inp.shape[0]
        tgt_len = tgt.shape[0]
        padded_inputs[i, :inp_len] = inp
        input_pad_mask[i, :inp_len] = False
        padded_targets[i, :tgt_len] = tgt
        target_mask[i, :tgt_len] = True

    return padded_inputs, input_pad_mask, padded_targets, target_mask


# ---------------------------------------------------------------------------
# Embedding Extraction
# ---------------------------------------------------------------------------

def extract_gt_embeddings(dataset_name, data_dir, split, grid_off, test_dataset=False):
    """
    Extract ReID embeddings for all ground-truth bounding boxes and group by track.

    Args:
        dataset_name: 'mot17', 'mot20', or 'dance'
        data_dir: Root data directory
        split: Annotation split name (e.g., 'train_half', 'train', 'val')
        grid_off: Whether to disable grid patches
        test_dataset: Whether this is a test dataset (affects model selection)

    Returns:
        List of np.ndarray, each with shape (T, emb_dim) or (T, 3, emb_dim)
    """
    # Determine paths
    if dataset_name == "mot17":
        direc = "mot"
        img_name = "train"
        ann_file = f"{split}.json"
    elif dataset_name == "mot20":
        direc = "MOT20"
        img_name = "train"
        ann_file = f"{split}.json"
    elif dataset_name == "dance":
        direc = "dancetrack"
        img_name = "train" if "train" in split else "val"
        ann_file = f"{split}.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    base_dir = os.path.join(data_dir, direc)
    ann_path = os.path.join(base_dir, "annotations", ann_file)

    print(f"Loading annotations from {ann_path}")
    with open(ann_path, "r") as f:
        coco_data = json.load(f)

    # Build mappings
    img_id_to_info = {img["id"]: img for img in coco_data["images"]}
    video_id_to_name = {v["id"]: v["file_name"] for v in coco_data.get("videos", [])}

    # Group annotations by (video_name, track_id) and sort by frame_id
    # Each entry: (frame_id, image_id, bbox, video_name)
    track_annotations = defaultdict(list)
    for ann in coco_data["annotations"]:
        if ann.get("category_id", 1) < 0:
            continue  # Skip ignored annotations
        img_info = img_id_to_info[ann["image_id"]]
        video_id = img_info["video_id"]
        video_name = video_id_to_name.get(video_id, f"video_{video_id}")
        frame_id = img_info["frame_id"]
        track_id = ann["track_id"]
        bbox = ann["bbox"]  # [x, y, w, h]
        # Convert to [x1, y1, x2, y2]
        x1, y1, w, h = bbox
        bbox_xyxy = [x1, y1, x1 + w, y1 + h]

        key = (video_name, track_id)
        track_annotations[key].append({
            "frame_id": frame_id,
            "image_id": ann["image_id"],
            "bbox": bbox_xyxy,
            "video_name": video_name,
        })

    # Sort each track by frame_id
    for key in track_annotations:
        track_annotations[key].sort(key=lambda x: x["frame_id"])

    print(f"Found {len(track_annotations)} tracks across "
          f"{len(set(k[0] for k in track_annotations))} videos")

    # Initialize embedding computer
    embedder = EmbeddingComputer(dataset_name, test_dataset, grid_off)

    # Cache path for extracted embeddings
    cache_dir = os.path.join("cache", "gt_embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{split}_{'basic' if grid_off else 'grid'}.pkl")

    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            track_sequences = pickle.load(f)
        print(f"Loaded {len(track_sequences)} track sequences")
        return track_sequences

    # Group annotations by image for batch processing
    image_annotations = defaultdict(list)
    for key, anns in track_annotations.items():
        for ann in anns:
            image_annotations[ann["image_id"]].append({
                "track_key": key,
                "bbox": ann["bbox"],
                "frame_idx_in_track": anns.index(ann),
            })

    # Process each image and extract embeddings
    track_embeddings = defaultdict(list)  # (video_name, track_id) -> list of (frame_id, embedding)
    image_ids = sorted(image_annotations.keys())

    print(f"Extracting embeddings for {len(image_ids)} images...")
    for img_idx, image_id in enumerate(image_ids):
        if img_idx % 100 == 0:
            print(f"  Processing image {img_idx + 1}/{len(image_ids)}")

        img_info = img_id_to_info[image_id]
        video_name = video_id_to_name.get(img_info["video_id"], f"video_{img_info['video_id']}")
        frame_id = img_info["frame_id"]
        file_name = img_info["file_name"]

        # Load image
        img_path = os.path.join(base_dir, img_name, file_name)
        if not os.path.exists(img_path):
            print(f"  WARNING: Image not found: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Failed to load image: {img_path}")
            continue

        # Collect all bboxes for this image
        anns_for_image = image_annotations[image_id]
        bboxes = np.array([a["bbox"] for a in anns_for_image], dtype=np.float32)

        # Compute embeddings (using a unique tag for caching)
        tag = f"gt_{video_name}:{frame_id}"
        embs = embedder.compute_embedding(img, bboxes, tag)
        # Shape: (N, emb_dim) or (N, 3, emb_dim) depending on grid mode

        # Assign embeddings to tracks
        for i, ann in enumerate(anns_for_image):
            key = ann["track_key"]
            track_embeddings[key].append((frame_id, embs[i]))

    # Sort each track's embeddings by frame_id and extract just the embeddings
    track_sequences = []
    for key in sorted(track_embeddings.keys()):
        emb_list = track_embeddings[key]
        emb_list.sort(key=lambda x: x[0])
        embeddings = np.stack([e for _, e in emb_list])
        track_sequences.append(embeddings)

    # Save cache
    embedder.dump_cache()
    print(f"Saving embeddings to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(track_sequences, f)

    print(f"Extracted {len(track_sequences)} track sequences")
    return track_sequences


def extract_gt_embeddings_kitti_mots(data_dir, split_seqs, grid_off,
                                      class_filter="all", split_name="train"):
    """
    Extract ReID embeddings for KITTI MOTS ground-truth detections with masks.

    Args:
        data_dir: Root data directory (contains kitti_mots/)
        split_seqs: List of sequence ID strings (e.g. ["0000", "0001", ...])
        grid_off: Whether to disable grid patches
        class_filter: "car" (class 1), "pedestrian" (class 2), or "all"
        split_name: Name for the split (used in cache filename)

    Returns:
        List of np.ndarray, each with shape (T, emb_dim) or (T, 3, emb_dim)
    """
    import pycocotools.mask as mask_util

    # Cache path
    cache_dir = os.path.join("cache", "gt_embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"kitti_mots_{split_name}_{'basic' if grid_off else 'grid'}.pkl",
    )

    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            track_sequences = pickle.load(f)
        print(f"Loaded {len(track_sequences)} track sequences")
        return track_sequences

    base_dir = os.path.join(data_dir, "kitti_mots", "training")
    ann_dir = os.path.join(base_dir, "instances_txt")
    img_dir = os.path.join(base_dir, "image_02")

    # Class ID mapping: 1=car, 2=pedestrian, 10=ignore
    class_ids = set()
    if class_filter == "car":
        class_ids = {1}
    elif class_filter == "pedestrian":
        class_ids = {2}
    else:  # "all"
        class_ids = {1, 2}

    # Parse annotations: group by (seq, object_id) → list of detections
    # Also group by (seq, frame_id) for batch processing
    track_annotations = defaultdict(list)  # (seq, obj_id) → [{frame_id, bbox, rle}]
    image_detections = defaultdict(list)   # (seq, frame_id) → [{obj_id, bbox, rle}]

    for seq in split_seqs:
        ann_path = os.path.join(ann_dir, f"{seq}.txt")
        if not os.path.exists(ann_path):
            print(f"  WARNING: Annotation file not found: {ann_path}")
            continue

        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                frame_id = int(parts[0])
                object_id = int(parts[1])
                class_id = int(parts[2])
                img_h = int(parts[3])
                img_w = int(parts[4])
                rle_string = parts[5]

                # Skip ignore regions
                if class_id == 10:
                    continue
                if class_id not in class_ids:
                    continue

                # Build RLE dict for pycocotools
                rle = {"size": [img_h, img_w], "counts": rle_string.encode("utf-8")}

                # Derive bbox: pycocotools toBbox returns [x, y, w, h]
                bbox_xywh = mask_util.toBbox(rle).tolist()
                x, y, w, h = bbox_xywh
                bbox_xyxy = [x, y, x + w, y + h]

                det = {
                    "frame_id": frame_id,
                    "bbox": bbox_xyxy,
                    "rle": rle,
                }
                track_annotations[(seq, object_id)].append(det)
                image_detections[(seq, frame_id)].append({
                    "track_key": (seq, object_id),
                    "bbox": bbox_xyxy,
                    "rle": rle,
                })

    # Sort each track by frame_id
    for key in track_annotations:
        track_annotations[key].sort(key=lambda x: x["frame_id"])

    print(f"Found {len(track_annotations)} tracks across "
          f"{len(split_seqs)} sequences (class_filter={class_filter})")

    # Initialize embedding computer
    # Use "mot17" as dataset for model selection (generic ReID)
    embedder = EmbeddingComputer("mot17", False, grid_off)

    # Process images and extract embeddings
    track_embeddings = defaultdict(list)  # (seq, obj_id) → [(frame_id, emb)]
    image_keys = sorted(image_detections.keys())

    print(f"Extracting embeddings for {len(image_keys)} images...")
    for img_idx, (seq, frame_id) in enumerate(image_keys):
        if img_idx % 100 == 0:
            print(f"  Processing image {img_idx + 1}/{len(image_keys)}")

        # Load image
        img_path = os.path.join(img_dir, seq, f"{frame_id:06d}.png")
        if not os.path.exists(img_path):
            print(f"  WARNING: Image not found: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Failed to load image: {img_path}")
            continue

        dets = image_detections[(seq, frame_id)]
        bboxes = np.array([d["bbox"] for d in dets], dtype=np.float32)

        # Decode RLE masks on-the-fly
        masks = []
        for d in dets:
            mask = mask_util.decode(d["rle"])  # (H, W) uint8
            masks.append(mask)

        tag = f"gt_kitti_{seq}:{frame_id}"
        embs = embedder.compute_embedding_masked(img, bboxes, masks, tag)

        for i, d in enumerate(dets):
            track_embeddings[d["track_key"]].append((frame_id, embs[i]))

    # Sort each track's embeddings by frame_id and stack
    track_sequences = []
    for key in sorted(track_embeddings.keys()):
        emb_list = track_embeddings[key]
        emb_list.sort(key=lambda x: x[0])
        embeddings = np.stack([e for _, e in emb_list])
        track_sequences.append(embeddings)

    # Save cache
    embedder.dump_cache()
    print(f"Saving embeddings to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(track_sequences, f)

    print(f"Extracted {len(track_sequences)} track sequences")
    return track_sequences


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def cosine_loss(predicted, target, mask=None):
    """
    Compute negative cosine similarity loss.

    Args:
        predicted: (B, T, D) predicted embeddings
        target:    (B, T, D) target embeddings
        mask:      (B, T) boolean mask, True for valid positions

    Returns:
        Scalar loss
    """
    # Normalize
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)

    # Cosine similarity per position
    cos_sim = (predicted * target).sum(dim=-1)  # (B, T)

    # Loss = 1 - cosine_similarity
    loss = 1.0 - cos_sim  # (B, T)

    if mask is not None:
        # Only count valid positions
        loss = loss * mask.float()
        n_valid = mask.float().sum()
        if n_valid > 0:
            loss = loss.sum() / n_valid
        else:
            loss = loss.sum()
    else:
        loss = loss.mean()

    return loss


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (inputs, pad_mask, targets, target_mask) in enumerate(dataloader):
        inputs = inputs.to(device)
        pad_mask = pad_mask.to(device)
        targets = targets.to(device)
        target_mask = target_mask.to(device)

        # Forward pass: model predicts next embedding at each position
        # Output shape: (B, T, emb_dim)
        output = model(inputs, pad_mask)

        # We want output at positions 0..T-2 to predict targets at positions 0..T-2
        # (targets are the shifted sequence: position i of target = position i+1 of input)
        # output[:, :-1, :] corresponds to predictions for positions 1..T-1
        pred = output[:, :-1, :]  # (B, T-1, emb_dim)

        # Trim predictions and targets to the same length
        min_len = min(pred.shape[1], targets.shape[1])
        pred = pred[:, :min_len, :]
        targets_trimmed = targets[:, :min_len, :]
        mask_trimmed = target_mask[:, :min_len]

        loss = cosine_loss(pred, targets_trimmed, mask_trimmed)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f})")

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    n_batches = 0
    n_valid = 0

    for inputs, pad_mask, targets, target_mask in dataloader:
        inputs = inputs.to(device)
        pad_mask = pad_mask.to(device)
        targets = targets.to(device)
        target_mask = target_mask.to(device)

        output = model(inputs, pad_mask)
        pred = output[:, :-1, :]

        min_len = min(pred.shape[1], targets.shape[1])
        pred = pred[:, :min_len, :]
        targets_trimmed = targets[:, :min_len, :]
        mask_trimmed = target_mask[:, :min_len]

        loss = cosine_loss(pred, targets_trimmed, mask_trimmed)
        total_loss += loss.item()

        # Compute average cosine similarity for valid positions
        pred_norm = F.normalize(pred, dim=-1)
        tgt_norm = F.normalize(targets_trimmed, dim=-1)
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # (B, T-1)
        valid = mask_trimmed.float()
        total_cos_sim += (cos_sim * valid).sum().item()
        n_valid += valid.sum().item()

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_cos_sim = total_cos_sim / max(n_valid, 1)
    return avg_loss, avg_cos_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser("Train GPT Appearance Predictor")

    # Data
    parser.add_argument("--dataset", type=str, default="mot17",
                        choices=["mot17", "mot20", "dance", "kitti_mots"],
                        help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Root data directory")
    parser.add_argument("--train_split", type=str, default=None,
                        help="Training annotation split (default: auto-select)")
    parser.add_argument("--val_split", type=str, default=None,
                        help="Validation annotation split (default: auto-select)")
    parser.add_argument("--grid_off", action="store_true",
                        help="Disable grid patches (use basic 512-dim embeddings)")
    parser.add_argument("--kitti_mots_class", type=str, default="all",
                        choices=["car", "pedestrian", "all"],
                        help="KITTI MOTS class filter (only used with --dataset kitti_mots)")

    # Model
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer hidden dimension")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of transformer blocks")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward hidden dimension")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--min_track_len", type=int, default=3,
                        help="Minimum track length to include in training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of tracks to use for validation")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/gpt_appearance",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine splits
    if args.dataset == "kitti_mots":
        # KITTI MOTS uses sequence-level splits, not annotation file splits
        if args.train_split is None:
            args.train_split = "train"
        if args.val_split is None:
            args.val_split = "val"
    else:
        if args.train_split is None:
            if args.dataset in ["mot17", "mot20"]:
                args.train_split = "train_half"
            elif args.dataset == "dance":
                args.train_split = "train"
        if args.val_split is None:
            if args.dataset in ["mot17", "mot20"]:
                args.val_split = "val_half"
            elif args.dataset == "dance":
                args.val_split = "val"

    # Determine embedding dimension
    emb_dim = 512 if args.grid_off else 512 * 3  # 1536 for grid mode

    print(f"\n{'='*60}")
    print(f"Training GPT Appearance Predictor")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Emb dim:    {emb_dim}")
    print(f"  Grid mode:  {'off' if args.grid_off else 'on'}")
    print(f"  d_model:    {args.d_model}")
    print(f"  n_heads:    {args.n_heads}")
    print(f"  n_layers:   {args.n_layers}")
    print(f"  d_ff:       {args.d_ff}")
    print(f"  max_seq:    {args.max_seq_len}")
    print(f"  epochs:     {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr:         {args.lr}")
    if args.dataset == "kitti_mots":
        print(f"  class:      {args.kitti_mots_class}")
    print(f"{'='*60}\n")

    # ---- Extract embeddings ----
    if args.dataset == "kitti_mots":
        # Resolve sequence lists for KITTI MOTS
        if args.train_split == "train":
            train_seqs = KITTI_MOTS_TRAIN_SEQS
        elif args.train_split == "val":
            train_seqs = KITTI_MOTS_VAL_SEQS
        else:
            train_seqs = args.train_split.split(",")

        if args.val_split == "val":
            val_seqs = KITTI_MOTS_VAL_SEQS
        elif args.val_split == "train":
            val_seqs = KITTI_MOTS_TRAIN_SEQS
        else:
            val_seqs = args.val_split.split(",")

        print("Extracting training embeddings...")
        train_sequences = extract_gt_embeddings_kitti_mots(
            args.data_dir, train_seqs, args.grid_off,
            class_filter=args.kitti_mots_class,
            split_name=args.train_split,
        )

        print("Extracting validation embeddings...")
        val_sequences = extract_gt_embeddings_kitti_mots(
            args.data_dir, val_seqs, args.grid_off,
            class_filter=args.kitti_mots_class,
            split_name=args.val_split,
        )
    else:
        print("Extracting training embeddings...")
        train_sequences = extract_gt_embeddings(
            args.dataset, args.data_dir, args.train_split, args.grid_off
        )

        # Split into train and validation
        np.random.seed(42)
        n_total = len(train_sequences)
        n_val = max(int(n_total * args.val_ratio), 1)
        indices = np.random.permutation(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        val_sequences = [train_sequences[i] for i in val_indices]
        train_sequences = [train_sequences[i] for i in train_indices]

    print(f"Train: {len(train_sequences)} tracks, Val: {len(val_sequences)} tracks")

    # ---- Create datasets ----
    train_dataset = TrackEmbeddingDataset(
        train_sequences, max_seq_len=args.max_seq_len, min_track_len=args.min_track_len
    )
    val_dataset = TrackEmbeddingDataset(
        val_sequences, max_seq_len=args.max_seq_len, min_track_len=args.min_track_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Create model ----
    model = AppearanceGPT(
        emb_dim=emb_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine annealing with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop ----
    best_val_loss = float("inf")
    config = {
        "emb_dim": emb_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "max_seq_len": args.max_seq_len,
        "dataset": args.dataset,
        "grid_off": args.grid_off,
    }

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_cos_sim = validate(model, val_loader, device)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val CosSim: {val_cos_sim:.4f} | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, f"best_{args.dataset}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_cos_sim": val_cos_sim,
                "config": config,
            }, save_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        # Periodic save
        if epoch % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}_{args.dataset}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_cos_sim": val_cos_sim,
                "config": config,
            }, save_path)
            print(f"  -> Saved checkpoint at epoch {epoch}")

    # Save final model
    save_path = os.path.join(args.output_dir, f"final_{args.dataset}.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_cos_sim": val_cos_sim,
        "config": config,
    }, save_path)
    print(f"\nTraining complete. Final model saved to {save_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
