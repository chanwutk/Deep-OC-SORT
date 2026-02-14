"""
Stage 2 fine-tuning script for the GPT-based appearance embedding predictor.

Fine-tunes the GPT model with prediction loss + contrastive loss, and evaluates
tracking performance via HOTA after each epoch using TrackEval.

Currently only fine-tunes the GPT (not ViT) since DINOv3 encoder is not yet
available. The contrastive loss is computed on pre-extracted embeddings and
logged for monitoring, but does not contribute gradients to the GPT.

Usage:
    python train_finetune.py --dataset mot20 \
        --gpt_checkpoint checkpoints/gpt_appearance/best_mot20.pth \
        --epochs 30 --batch_size 16
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import dataset
import utils
from external.adaptors import detector
from trackers import integrated_ocsort_embedding as tracker_module
from trackers.integrated_ocsort_embedding.gpt_model import AppearanceGPT, AppearanceTracker

# Reuse embedding extraction and loss from Stage 1
from train_gpt import (
    extract_gt_embeddings, cosine_loss,
    extract_gt_embeddings_kitti_mots,
    KITTI_MOTS_TRAIN_SEQS, KITTI_MOTS_VAL_SEQS,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FinetuneTrackDataset(Dataset):
    """
    Dataset for Stage 2 fine-tuning with contrastive triplet sampling.

    Each sample provides:
        - input_seq / target_seq: same as Stage 1 (for prediction loss)
        - anchor, positive, hard_neg, neg: triplets (for contrastive loss)
    """

    def __init__(self, track_sequences, max_seq_len=256, min_track_len=3):
        self.max_seq_len = max_seq_len
        self.samples = []       # (chunk, track_idx) pairs
        self.track_indices = [] # which original track each sample came from
        self.all_embeddings = []  # flat list of (embedding, track_idx) for negative sampling

        for track_idx, seq in enumerate(track_sequences):
            if len(seq) < min_track_len:
                continue

            # Flatten grid embeddings: (T, 3, 512) -> (T, 1536)
            if seq.ndim == 3:
                T, G, D = seq.shape
                seq = seq.reshape(T, G * D)

            # Collect all embeddings for negative sampling
            for t in range(len(seq)):
                self.all_embeddings.append((seq[t], track_idx))

            # Split long sequences into overlapping chunks
            if len(seq) <= max_seq_len:
                self.samples.append((seq, track_idx))
            else:
                stride = max(max_seq_len // 2, 1)
                for start in range(0, len(seq) - min_track_len + 1, stride):
                    end = min(start + max_seq_len, len(seq))
                    chunk = seq[start:end]
                    if len(chunk) >= min_track_len:
                        self.samples.append((chunk, track_idx))

        # Build per-track embedding index for faster negative sampling
        self._track_emb_indices = defaultdict(list)
        for i, (_, tidx) in enumerate(self.all_embeddings):
            self._track_emb_indices[tidx].append(i)
        self._all_track_indices = list(self._track_emb_indices.keys())

        print(f"FinetuneTrackDataset: {len(self.samples)} samples, "
              f"{len(self.all_embeddings)} total embeddings from "
              f"{len(self._all_track_indices)} tracks")

    def __len__(self):
        return len(self.samples)

    def _sample_negative(self, exclude_track_idx):
        """Sample an embedding from a different track."""
        candidates = [t for t in self._all_track_indices if t != exclude_track_idx]
        if not candidates:
            # Only one track — return a random embedding
            idx = np.random.randint(len(self.all_embeddings))
            return self.all_embeddings[idx][0]
        neg_track = candidates[np.random.randint(len(candidates))]
        emb_indices = self._track_emb_indices[neg_track]
        idx = emb_indices[np.random.randint(len(emb_indices))]
        return self.all_embeddings[idx][0]

    def __getitem__(self, idx):
        seq, track_idx = self.samples[idx]
        T = len(seq)

        # For prediction loss (same as Stage 1)
        input_seq = torch.from_numpy(seq).float()        # (T, emb_dim)
        target_seq = torch.from_numpy(seq[1:]).float()    # (T-1, emb_dim)

        # For contrastive loss: sample triplets
        t = np.random.randint(0, T - 1)
        anchor = torch.from_numpy(seq[t]).float()
        positive = torch.from_numpy(seq[t + 1]).float()

        # Hard negative: same track, non-consecutive
        hard_neg_candidates = [i for i in range(T) if i != t and i != t + 1]
        if hard_neg_candidates:
            k = hard_neg_candidates[np.random.randint(len(hard_neg_candidates))]
            hard_neg = torch.from_numpy(seq[k]).float()
        else:
            hard_neg = anchor.clone()  # fallback

        # Negative: different track
        neg_emb = self._sample_negative(track_idx)
        neg = torch.from_numpy(neg_emb).float()

        return input_seq, target_seq, anchor, positive, hard_neg, neg


def collate_finetune_fn(batch):
    """Custom collate for variable-length sequences + fixed-size triplets."""
    input_seqs, target_seqs, anchors, positives, hard_negs, negs = zip(*batch)

    # Pad input/target sequences (same logic as train_gpt.py collate_fn)
    max_input_len = max(s.shape[0] for s in input_seqs)
    max_target_len = max(s.shape[0] for s in target_seqs)
    emb_dim = input_seqs[0].shape[1]
    batch_size = len(input_seqs)

    padded_inputs = torch.zeros(batch_size, max_input_len, emb_dim)
    input_pad_mask = torch.ones(batch_size, max_input_len, dtype=torch.bool)

    padded_targets = torch.zeros(batch_size, max_target_len, emb_dim)
    target_mask = torch.zeros(batch_size, max_target_len, dtype=torch.bool)

    for i, (inp, tgt) in enumerate(zip(input_seqs, target_seqs)):
        inp_len = inp.shape[0]
        tgt_len = tgt.shape[0]
        padded_inputs[i, :inp_len] = inp
        input_pad_mask[i, :inp_len] = False
        padded_targets[i, :tgt_len] = tgt
        target_mask[i, :tgt_len] = True

    # Stack triplets (all same size)
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    hard_negs = torch.stack(hard_negs)
    negs = torch.stack(negs)

    return (padded_inputs, input_pad_mask, padded_targets, target_mask,
            anchors, positives, hard_negs, negs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    lambda_contrastive, triplet_margin):
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_contr_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        (inputs, pad_mask, targets, target_mask,
         anchors, positives, hard_negs, negs) = batch

        inputs = inputs.to(device)
        pad_mask = pad_mask.to(device)
        targets = targets.to(device)
        target_mask = target_mask.to(device)
        anchors = anchors.to(device)
        positives = positives.to(device)
        hard_negs = hard_negs.to(device)
        negs = negs.to(device)

        # Prediction loss (same as Stage 1)
        output = model(inputs, pad_mask)
        pred = output[:, :-1, :]

        min_len = min(pred.shape[1], targets.shape[1])
        pred = pred[:, :min_len, :]
        targets_trimmed = targets[:, :min_len, :]
        mask_trimmed = target_mask[:, :min_len]

        loss_pred = cosine_loss(pred, targets_trimmed, mask_trimmed)

        # Contrastive loss (triplet margin)
        # With frozen embeddings this has no gradient w.r.t. model params,
        # but we compute and log it for monitoring. It becomes active when
        # ViT fine-tuning is enabled.
        loss_hard = F.triplet_margin_loss(
            anchors, positives, hard_negs, margin=triplet_margin)
        loss_diff = F.triplet_margin_loss(
            anchors, positives, negs, margin=triplet_margin)
        loss_contrastive = 0.5 * (loss_hard + loss_diff)

        loss = loss_pred + lambda_contrastive * loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_pred_loss += loss_pred.item()
        total_contr_loss += loss_contrastive.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg = total_loss / n_batches
            print(f"  Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} (avg: {avg:.4f})")

    n = max(n_batches, 1)
    return total_loss / n, total_pred_loss / n, total_contr_loss / n


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    n_batches = 0
    n_valid = 0

    for batch in dataloader:
        (inputs, pad_mask, targets, target_mask,
         anchors, positives, hard_negs, negs) = batch

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

        pred_norm = F.normalize(pred, dim=-1)
        tgt_norm = F.normalize(targets_trimmed, dim=-1)
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)
        valid = mask_trimmed.float()
        total_cos_sim += (cos_sim * valid).sum().item()
        n_valid += valid.sum().item()

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_cos_sim = total_cos_sim / max(n_valid, 1)
    return avg_loss, avg_cos_sim


# ---------------------------------------------------------------------------
# Tracking Inference
# ---------------------------------------------------------------------------

def run_tracking(dataset_name, gpt_model, result_folder, exp_name,
                 test_dataset=False, grid_off=True, gpt_off=False,
                 w_assoc_emb=0.75, alpha_fixed_emb=0.95, data_dir="datasets"):
    """Run full tracking inference and save results in MOT format.

    Replicates the inference loop from main.py as a self-contained function.
    """
    # Detector paths (same as main.py)
    if dataset_name == "mot17":
        if test_dataset:
            detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
        else:
            detector_path = "external/weights/bytetrack_ablation.pth.tar"
        size = (800, 1440)
    elif dataset_name == "mot20":
        if test_dataset:
            detector_path = "external/weights/bytetrack_x_mot20.tar"
            size = (896, 1600)
        else:
            detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
            size = (800, 1440)
    elif dataset_name == "dance":
        detector_path = "external/weights/bytetrack_dance_model.pth.tar"
        size = (800, 1440)
    else:
        raise RuntimeError(f"Unknown dataset: {dataset_name}")

    det = detector.Detector("yolox", detector_path, dataset_name)
    loader = dataset.get_mot_loader(dataset_name, test_dataset, data_dir=data_dir, size=size)

    # Set GPT model directly (no file I/O)
    if not gpt_off and gpt_model is not None:
        AppearanceTracker.set_model(gpt_model, grid_off=grid_off)
    else:
        AppearanceTracker.set_model(None, grid_off=grid_off)

    # Build a minimal args namespace for OCSort
    tracker_args_parser = tracker_module.args.make_parser()
    tracker_args = tracker_args_parser.parse_args([])
    tracker_args.dataset = dataset_name
    tracker_args.test_dataset = test_dataset

    oc_sort_args = dict(
        args=tracker_args,
        det_thresh=tracker_args.track_thresh,
        iou_threshold=tracker_args.iou_thresh,
        asso_func=tracker_args.asso,
        delta_t=tracker_args.deltat,
        inertia=tracker_args.inertia,
        w_association_emb=w_assoc_emb,
        alpha_fixed_emb=alpha_fixed_emb,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        aw_param=0.5,
        new_kf_off=False,
        grid_off=grid_off,
        gpt_model_path=None,  # model already set via set_model
        gpt_off=gpt_off,
    )

    tracker = tracker_module.ocsort.OCSort(**oc_sort_args)
    results = {}

    for (img, np_img), label, info, idx in loader:
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]

        if "FRCNN" not in video_name and dataset_name == "mot17":
            continue

        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []

        img = img.cuda()

        if frame_id == 1:
            tracker.dump_cache()
            tracker = tracker_module.ocsort.OCSort(**oc_sort_args)

        pred = det(img, tag)
        if pred is None:
            continue

        targets = tracker.update(pred, img, np_img[0].numpy(), tag)
        tlwhs, ids = utils.filter_targets(targets, aspect_ratio_thresh=1.6, min_box_area=10)
        results[video_name].append((frame_id, tlwhs, ids))

    tracker.dump_cache()

    # Save results
    folder = os.path.join(result_folder, exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)

    return os.path.join(result_folder, exp_name)


# ---------------------------------------------------------------------------
# HOTA Evaluation
# ---------------------------------------------------------------------------

def compute_hota(result_folder, gt_folder, benchmark, split, exp_name):
    """Compute HOTA score using TrackEval Python API."""
    sys.path.insert(0, "external/TrackEval")
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_ONLY_COMBINED"] = True
    eval_config["OUTPUT_SUMMARY"] = False
    eval_config["OUTPUT_DETAILED"] = False
    eval_config["PLOT_CURVES"] = False
    eval_config["DISPLAY_LESS_PROGRESS"] = True

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = gt_folder
    dataset_config["TRACKERS_FOLDER"] = result_folder
    dataset_config["BENCHMARK"] = benchmark
    dataset_config["SPLIT_TO_EVAL"] = split
    dataset_config["TRACKERS_TO_EVAL"] = [exp_name]
    dataset_config["TRACKER_SUB_FOLDER"] = "data"

    evaluator = trackeval.Evaluator(eval_config)
    ds = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA()]
    output_res, output_msg = evaluator.evaluate([ds], metrics)

    # Extract HOTA: average across alpha thresholds
    tracker_res = output_res["MotChallenge2DBox"][exp_name]
    combined = tracker_res["COMBINED_SEQ"]
    hota_array = combined["cls_comb_cls_av"]["HOTA"]["HOTA"]
    hota_score = np.mean(hota_array)

    deta = np.mean(combined["cls_comb_cls_av"]["HOTA"]["DetA"])
    assa = np.mean(combined["cls_comb_cls_av"]["HOTA"]["AssA"])

    return hota_score, deta, assa


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser("Stage 2: Fine-tune GPT Appearance Predictor")

    # Data
    parser.add_argument("--dataset", type=str, default="mot20",
                        choices=["mot17", "mot20", "dance", "kitti_mots"],
                        help="Dataset for training embeddings")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Root data directory")
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--grid_off", action="store_true",
                        help="Disable grid patches")
    parser.add_argument("--kitti_mots_class", type=str, default="all",
                        choices=["car", "pedestrian", "all"],
                        help="KITTI MOTS class filter (only used with --dataset kitti_mots)")

    # GPT model (from Stage 1)
    parser.add_argument("--gpt_checkpoint", type=str, required=True,
                        help="Path to Stage 1 GPT checkpoint")

    # Fine-tuning hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_gpt", type=float, default=1e-4,
                        help="Learning rate for GPT")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lambda_contrastive", type=float, default=0.1,
                        help="Weight for contrastive loss")
    parser.add_argument("--triplet_margin", type=float, default=0.3,
                        help="Margin for triplet loss")
    parser.add_argument("--min_track_len", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # Tracking evaluation
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Dataset for HOTA evaluation (default: same as --dataset)")
    parser.add_argument("--gt_folder", type=str, default="results/gt/",
                        help="Ground truth folder for TrackEval")
    parser.add_argument("--result_folder", type=str, default="results/trackers/",
                        help="Result folder for tracking output")

    # Tracker args for inference
    parser.add_argument("--w_assoc_emb", type=float, default=0.75)
    parser.add_argument("--alpha_fixed_emb", type=float, default=0.95)

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetune")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save periodic checkpoint every N epochs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine splits
    if args.dataset == "kitti_mots":
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

    # Evaluation dataset defaults to training dataset
    eval_dataset = args.eval_dataset or args.dataset

    # Whether tracking evaluation is available
    # KITTI MOTS has no YOLOX detector, so skip tracking eval
    tracking_eval_available = eval_dataset != "kitti_mots"

    if tracking_eval_available:
        # Determine benchmark name and split for TrackEval
        benchmark_map = {"mot17": "MOT17", "mot20": "MOT20", "dance": "DANCE"}
        benchmark = benchmark_map[eval_dataset]
        eval_split = "val"

        # Result folder for tracking
        result_folder_base = os.path.join(args.result_folder, f"{benchmark}-{eval_split}")
    else:
        benchmark = None
        eval_split = None
        result_folder_base = None

    # ---- Load Stage 1 GPT checkpoint ----
    print(f"Loading GPT checkpoint from {args.gpt_checkpoint}")
    checkpoint = torch.load(args.gpt_checkpoint, map_location=device)
    config = checkpoint["config"]

    emb_dim = config["emb_dim"]
    grid_off = config.get("grid_off", args.grid_off)

    model = AppearanceGPT(
        emb_dim=config["emb_dim"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=0.1,  # keep dropout during fine-tuning
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded GPT from epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', '?')}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\n{'='*60}")
    print(f"Stage 2: Fine-tune GPT Appearance Predictor")
    print(f"  Dataset:             {args.dataset}")
    print(f"  Eval dataset:        {eval_dataset}")
    print(f"  Emb dim:             {emb_dim}")
    print(f"  Grid mode:           {'off' if grid_off else 'on'}")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  LR (GPT):           {args.lr_gpt}")
    print(f"  Lambda contrastive:  {args.lambda_contrastive}")
    print(f"  Triplet margin:      {args.triplet_margin}")
    print(f"{'='*60}\n")

    # ---- Extract embeddings ----
    if args.dataset == "kitti_mots":
        # Resolve sequence lists
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
        train_sequences_split = extract_gt_embeddings_kitti_mots(
            args.data_dir, train_seqs, grid_off,
            class_filter=args.kitti_mots_class,
            split_name=args.train_split,
        )

        print("Extracting validation embeddings...")
        val_sequences = extract_gt_embeddings_kitti_mots(
            args.data_dir, val_seqs, grid_off,
            class_filter=args.kitti_mots_class,
            split_name=args.val_split,
        )
    else:
        print("Extracting training embeddings...")
        train_sequences = extract_gt_embeddings(
            args.dataset, args.data_dir, args.train_split, grid_off
        )

        # Split into train and validation
        np.random.seed(42)
        n_total = len(train_sequences)
        n_val = max(int(n_total * args.val_ratio), 1)
        indices = np.random.permutation(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        val_sequences = [train_sequences[i] for i in val_indices]
        train_sequences_split = [train_sequences[i] for i in train_indices]

    print(f"Train: {len(train_sequences_split)} tracks, Val: {len(val_sequences)} tracks")

    # ---- Create datasets ----
    train_dataset = FinetuneTrackDataset(
        train_sequences_split, max_seq_len=args.max_seq_len,
        min_track_len=args.min_track_len,
    )
    val_dataset = FinetuneTrackDataset(
        val_sequences, max_seq_len=args.max_seq_len,
        min_track_len=args.min_track_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_finetune_fn, num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_finetune_fn, num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr_gpt, weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop ----
    best_hota = -1.0
    best_val_loss = float("inf")
    finetune_config = {
        **config,
        "stage": 2,
        "lambda_contrastive": args.lambda_contrastive,
        "triplet_margin": args.triplet_margin,
        "lr_gpt": args.lr_gpt,
    }

    exp_name = "finetune"  # single exp_name, overwritten each epoch

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_pred_loss, train_contr_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            args.lambda_contrastive, args.triplet_margin,
        )
        val_loss, val_cos_sim = validate(model, val_loader, device)

        hota, deta, assa = 0.0, 0.0, 0.0
        tracking_time = 0.0

        if tracking_eval_available:
            # Run HOTA evaluation
            model.eval()
            print(f"\n  Running tracking evaluation (epoch {epoch})...")
            tracking_start = time.time()
            run_tracking(
                eval_dataset, model, result_folder_base, exp_name,
                test_dataset=False, grid_off=grid_off, gpt_off=False,
                w_assoc_emb=args.w_assoc_emb, alpha_fixed_emb=args.alpha_fixed_emb,
                data_dir=args.data_dir,
            )
            hota, deta, assa = compute_hota(
                result_folder_base, args.gt_folder, benchmark, eval_split, exp_name,
            )
            tracking_time = time.time() - tracking_start

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        if tracking_eval_available:
            print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s, tracking: {tracking_time:.1f}s) | "
                  f"Train Loss: {train_loss:.4f} (pred: {train_pred_loss:.4f}, "
                  f"contr: {train_contr_loss:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | Val CosSim: {val_cos_sim:.4f} | "
                  f"HOTA: {hota:.4f} | DetA: {deta:.4f} | AssA: {assa:.4f} | "
                  f"LR: {current_lr:.6f}")
        else:
            print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
                  f"Train Loss: {train_loss:.4f} (pred: {train_pred_loss:.4f}, "
                  f"contr: {train_contr_loss:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | Val CosSim: {val_cos_sim:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Checkpoint: use HOTA if available, otherwise use val_loss
        save_best = False
        if tracking_eval_available:
            if hota > best_hota:
                best_hota = hota
                save_best = True
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best = True

        if save_best:
            save_path = os.path.join(args.output_dir, f"best_{args.dataset}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hota": hota,
                "deta": deta,
                "assa": assa,
                "val_loss": val_loss,
                "val_cos_sim": val_cos_sim,
                "config": finetune_config,
            }, save_path)
            if tracking_eval_available:
                print(f"  -> Saved best model (HOTA={hota:.4f})")
            else:
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        # Periodic save
        if epoch % args.save_every == 0:
            save_path = os.path.join(
                args.output_dir, f"epoch_{epoch}_{args.dataset}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hota": hota,
                "deta": deta,
                "assa": assa,
                "val_loss": val_loss,
                "val_cos_sim": val_cos_sim,
                "config": finetune_config,
            }, save_path)
            print(f"  -> Saved checkpoint at epoch {epoch}")

    # Save final model
    save_path = os.path.join(args.output_dir, f"final_{args.dataset}.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hota": hota,
        "val_loss": val_loss,
        "val_cos_sim": val_cos_sim,
        "config": finetune_config,
    }, save_path)
    print(f"\nTraining complete. Final model saved to {save_path}")
    if tracking_eval_available:
        print(f"Best HOTA: {best_hota:.4f}")
    else:
        print(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
