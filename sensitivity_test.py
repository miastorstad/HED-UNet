#!/usr/bin/env python3
"""
Sensitivity test script for image distortion effects.

Example:
    python sensitivity_test.py --run-dir logs/2026-01-20_13-40-33 \
        --checkpoint 99 --distortion gaussian_noise --levels 0,0.01,0.02
"""
import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF
import yaml
import matplotlib.pyplot as plt

from data_loading import get_dataset
from deep_learning import get_model, get_loss


def parse_levels(raw):
    if raw is None:
        return []
    return [float(x) for x in raw.split(",") if x.strip() != ""]


def find_latest_run(logs_dir):
    if not logs_dir.exists():
        return None
    candidates = [p for p in logs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_checkpoint(run_dir, checkpoint):
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path, checkpoint_path.stem
    if checkpoint in ("best_f1", "best"):
        name = "best_f1.pt" if checkpoint == "best_f1" else "best.pt"
        return run_dir / "checkpoints" / name, checkpoint
    if checkpoint.isdigit():
        epoch = int(checkpoint)
        return run_dir / "checkpoints" / f"{epoch:02d}.pt", f"epoch{epoch}"
    raise ValueError(f"Unrecognized checkpoint value: {checkpoint}")


def parse_seg_threshold(metrics_path, epoch):
    if not metrics_path.exists():
        return None
    pattern = re.compile(rf"Epoch {epoch:02d} - Val: .*SegThreshold: ([0-9.]+)")
    best = None
    with metrics_path.open("r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                best = float(match.group(1))
    return best


def parse_test_metrics(metrics_path):
    if not metrics_path.exists():
        return {}
    test_line = None
    with metrics_path.open("r") as f:
        for line in f:
            if line.startswith("Test:"):
                test_line = line.strip()
    if not test_line:
        return {}
    metrics = {}
    parts = test_line.replace("Test:", "").strip().split(",")
    for part in parts:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip()
        val = val.strip()
        try:
            metrics[key] = float(val)
        except ValueError:
            continue
    return metrics


def load_split_indices(dataset, split_file):
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not hasattr(dataset, "tiles"):
        raise ValueError("Dataset has no tiles attribute; cannot map split file.")
    tile_map = {str(path): idx for idx, path in enumerate(dataset.tiles)}
    indices = []
    missing = []
    with split_file.open("r") as f:
        for line in f:
            tile_path = line.strip()
            if not tile_path:
                continue
            idx = tile_map.get(tile_path)
            if idx is None:
                missing.append(tile_path)
            else:
                indices.append(idx)
    if missing:
        missing_preview = ", ".join(missing[:5])
        raise ValueError(
            f"Split file contains {len(missing)} unknown tiles. "
            f"Examples: {missing_preview}"
        )
    return indices


class DistortedDataset(Dataset):
    def __init__(self, dataset, distortion, level, base_resolution=0.2):
        self.dataset = dataset
        self.distortion = distortion
        self.level = float(level)
        self.base_resolution = float(base_resolution)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.distortion == "none" or self.level == 0.0:
            return img, target

        img = img.clone()
        if self.distortion == "gaussian_noise":
            img = img + torch.randn_like(img) * self.level
            img = torch.clamp(img, 0.0, 1.0)
        elif self.distortion == "gaussian_blur":
            sigma = max(0.0, self.level)
            if sigma > 0:
                kernel = int(max(3, 2 * round(3 * sigma) + 1))
                img = TF.gaussian_blur(img, kernel_size=kernel, sigma=sigma)
        elif self.distortion == "brightness":
            factor = 1.0 + self.level
            img = torch.clamp(img * factor, 0.0, 1.0)
        elif self.distortion == "contrast":
            factor = 1.0 + self.level
            mean = img.mean(dim=(1, 2), keepdim=True)
            img = torch.clamp((img - mean) * factor + mean, 0.0, 1.0)
        elif self.distortion == "resolution":
            target_res = self.level
            if target_res <= 0:
                raise ValueError("Resolution level must be > 0 (meters/pixel).")
            scale = self.base_resolution / target_res
            if scale <= 0:
                raise ValueError("Computed scale must be > 0.")
            if abs(scale - 1.0) < 1e-6:
                return img, target
            c, h, w = img.shape
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            resized = F.interpolate(
                img.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            img = F.interpolate(
                resized,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            raise ValueError(f"Unknown distortion: {self.distortion}")

        return img, target


def compute_weighted_loss(y_hat, target, loss_fn, weights):
    if y_hat.shape[1] < 2 or target.shape[1] < 2:
        return loss_fn(y_hat, target)
    seg_weight = weights.get("seg", 1.0)
    edge_weight = weights.get("edge", 1.0)
    seg_loss = loss_fn(y_hat[:, 1], target[:, 0])
    edge_loss = loss_fn(y_hat[:, 0], target[:, 1])
    return seg_weight * seg_loss + edge_weight * edge_loss


def get_pyramid(mask, stack_height, sobel, model_name):
    with torch.no_grad():
        masks = [mask]
        for _ in range(stack_height):
            big_mask = masks[-1]
            small_mask = F.avg_pool2d(big_mask, 2)
            masks.append(small_mask)

        targets = []
        for m in masks:
            sobel_edges = torch.any(sobel(m) != 0, dim=1, keepdims=True).float()
            if model_name == "HED":
                targets.append(sobel_edges)
            else:
                targets.append(torch.cat([m, sobel_edges], dim=1))
    return targets


def compute_binary_classification_metrics(tp, fp, fn, tn, eps=1e-6):
    precision_pos = tp / (tp + fp + eps)
    recall_pos = tp / (tp + fn + eps)
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + eps)

    precision_neg = tn / (tn + fn + eps)
    recall_neg = tn / (tn + fp + eps)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + eps)

    support_pos = tp + fn
    support_neg = tn + fp
    total = support_pos + support_neg + eps

    macro_precision = (precision_pos + precision_neg) / 2
    macro_recall = (recall_pos + recall_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2

    weighted_precision = ((precision_pos * support_pos) + (precision_neg * support_neg)) / total
    weighted_recall = ((recall_pos * support_pos) + (recall_neg * support_neg)) / total
    weighted_f1 = ((f1_pos * support_pos) + (f1_neg * support_neg)) / total

    specificity_pos = recall_neg
    specificity_neg = recall_pos
    macro_specificity = (specificity_pos + specificity_neg) / 2
    balanced_accuracy = (recall_pos + recall_neg) / 2
    macro_accuracy = (recall_pos + recall_neg) / 2

    def cohen_kappa(tp_k, fp_k, fn_k, tn_k):
        total_k = tp_k + fp_k + fn_k + tn_k + eps
        po_k = (tp_k + tn_k) / total_k
        pe_k = ((tp_k + fp_k) * (tp_k + fn_k) + (fn_k + tn_k) * (fp_k + tn_k)) / (total_k * total_k)
        return (po_k - pe_k) / (1 - pe_k + eps)

    kappa_pos = cohen_kappa(tp, fp, fn, tn)
    kappa_neg = cohen_kappa(tn, fn, fp, tp)
    macro_kappa = (kappa_pos + kappa_neg) / 2

    return dict(
        SegF1Macro=float(macro_f1),
        SegF1Weighted=float(weighted_f1),
        SegPrecisionMacro=float(macro_precision),
        SegPrecisionWeighted=float(weighted_precision),
        SegRecallMacro=float(macro_recall),
        SegRecallWeighted=float(weighted_recall),
        SegSpecificity=float(macro_specificity),
        SegBalancedAcc=float(balanced_accuracy),
        SegMacroAcc=float(macro_accuracy),
        SegCohenKappa=float(macro_kappa),
    )


@torch.no_grad()
def evaluate(model, loader, loss_fn, loss_weights, seg_threshold, stack_height, sobel, model_name, device):
    model.eval()
    loss_sum = 0.0
    batch_count = 0

    seg_tp = seg_fp = seg_fn = seg_tn = 0.0
    seg_pos_true_sum = 0.0
    seg_pos_pred_sum = 0.0
    edge_correct = 0.0
    edge_total = 0.0

    for img, target in loader:
        img = img.to(device)
        target = target.to(device)

        y_hat, y_hat_levels = model(img)
        targets = get_pyramid(target, stack_height, sobel, model_name)
        loss_levels = []
        if y_hat_levels:
            for y_hat_el, t in zip(y_hat_levels, targets):
                loss_levels.append(compute_weighted_loss(y_hat_el, t, loss_fn, loss_weights))
        loss_final = compute_weighted_loss(y_hat, targets[0], loss_fn, loss_weights)
        loss_deep_super = torch.sum(torch.stack(loss_levels)) if loss_levels else torch.zeros(1, device=device)
        loss = loss_final + loss_deep_super

        loss_sum += float(loss.item())
        batch_count += 1

        target0 = targets[0]
        seg_logit = y_hat[:, 1]
        seg_prob = torch.sigmoid(seg_logit)
        seg_pred = (seg_prob > seg_threshold).float()
        seg_true = target0[:, 0]

        seg_tp += float(((seg_pred == 1) & (seg_true == 1)).sum().item())
        seg_fp += float(((seg_pred == 1) & (seg_true == 0)).sum().item())
        seg_fn += float(((seg_pred == 0) & (seg_true == 1)).sum().item())
        seg_tn += float(((seg_pred == 0) & (seg_true == 0)).sum().item())
        seg_pos_true_sum += float(seg_true.sum().item())
        seg_pos_pred_sum += float(seg_pred.sum().item())

        edge_logit = y_hat[:, 0]
        edge_pred = (torch.sigmoid(edge_logit) > 0.5).float()
        edge_true = target0[:, 1]
        edge_correct += float((edge_pred == edge_true).sum().item())
        edge_total += float(edge_true.numel())

    total = seg_tp + seg_fp + seg_fn + seg_tn + 1e-6
    seg_acc = (seg_tp + seg_tn) / total
    seg_pos_true = seg_pos_true_sum / total
    seg_pos_pred = seg_pos_pred_sum / total
    seg_recall = seg_tp / (seg_tp + seg_fn + 1e-6)
    seg_precision = seg_tp / (seg_tp + seg_fp + 1e-6)
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall + 1e-6)
    edge_acc = edge_correct / (edge_total + 1e-6)

    metrics_vals = dict(
        Loss=(loss_sum / max(batch_count, 1)),
        SegAcc=seg_acc,
        EdgeAcc=edge_acc,
        SegPosTrue=seg_pos_true,
        SegPosPred=seg_pos_pred,
        SegRecall=seg_recall,
        SegPrecision=seg_precision,
        SegF1=seg_f1,
        SegThreshold=float(seg_threshold),
    )
    metrics_vals.update(compute_binary_classification_metrics(seg_tp, seg_fp, seg_fn, seg_tn))
    return metrics_vals


def main():
    parser = argparse.ArgumentParser(description="Sensitivity test for image distortions.")
    parser.add_argument("--run-dir", default=None, help="Path to run dir in logs.")
    parser.add_argument("--checkpoint", default="99", help="Epoch number, 'best_f1', or a .pt path.")
    parser.add_argument("--split-file", default=None, help="Split file to use (default: test_split.txt in run dir).")
    parser.add_argument("--distortion", default="gaussian_noise",
                        choices=["none", "gaussian_noise", "gaussian_blur", "brightness", "contrast", "resolution"])
    parser.add_argument("--levels", default="0,0.01,0.02,0.05,0.1",
                        help="Comma-separated distortion levels.")
    parser.add_argument("--base-resolution", type=float, default=0.2,
                        help="Base resolution in meters/pixel for resolution tests.")
    parser.add_argument("--seg-threshold", type=float, default=None,
                        help="Segmentation threshold (defaults to threshold from metrics.txt if available).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--data-threads", type=int, default=None, help="Override data loader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--plot", action="store_true", help="Generate plots for each metric.")
    args = parser.parse_args()

    logs_dir = Path("logs")
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(logs_dir)
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError("Could not locate run directory. Provide --run-dir.")

    config_path = run_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yml in {run_dir}")
    config = yaml.load(config_path.open(), Loader=yaml.SafeLoader)

    checkpoint_path, checkpoint_label = resolve_checkpoint(run_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    epoch_num = int(args.checkpoint) if args.checkpoint.isdigit() else None
    seg_threshold = args.seg_threshold
    if seg_threshold is None and epoch_num is not None:
        seg_threshold = parse_seg_threshold(run_dir / "metrics.txt", epoch_num)
    if seg_threshold is None:
        seg_threshold = 0.5

    split_file = Path(args.split_file) if args.split_file else run_dir / "test_split.txt"
    levels = parse_levels(args.levels)
    if not levels:
        raise ValueError("No distortion levels provided.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    modelclass = get_model(config["model"])
    model = modelclass(**config["model_args"])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    stack_height = config.get("model_args", {}).get("stack_height", 1)
    loss_weights = config.get("loss_weights", {})
    loss_function = get_loss(config["loss_args"])
    if isinstance(loss_function, torch.nn.Module):
        loss_function = loss_function.to(device)

    sobel = nn.Conv2d(1, 2, 3, padding=1, padding_mode="replicate", bias=False)
    sobel.weight.requires_grad = False
    sobel.weight.set_(torch.Tensor([[
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]],
       [[-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]]).reshape(2, 1, 3, 3))
    sobel = sobel.to(device)

    dataset = get_dataset("train")
    split_indices = load_split_indices(dataset, split_file)
    base_dataset = Subset(dataset, split_indices)

    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    data_threads = args.data_threads if args.data_threads is not None else config["data_threads"]

    results = []
    for level in levels:
        distorted_dataset = DistortedDataset(
            base_dataset, args.distortion, level, base_resolution=args.base_resolution
        )
        loader = DataLoader(
            distorted_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_threads,
            pin_memory=True,
        )
        metrics = evaluate(
            model,
            loader,
            loss_function,
            loss_weights,
            seg_threshold,
            stack_height,
            sobel,
            config["model"],
            device,
        )
        row = dict(
            distortion=args.distortion,
            level=level,
            checkpoint=checkpoint_label,
            run_dir=str(run_dir),
            split_file=str(split_file),
        )
        row.update(metrics)
        results.append(row)
        summary = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k in ("Loss", "SegAcc", "SegF1", "EdgeAcc"))
        print(f"Level {level:.4f} -> {summary}")

    output_path = Path(args.output) if args.output else (
        run_dir / "sensitivity" / f"{args.distortion}_{checkpoint_label}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved results to {output_path}")

    if args.plot:
        base_metrics = parse_test_metrics(run_dir / "metrics.txt")
        plot_dir = output_path.parent / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        levels_sorted = [row["level"] for row in results]
        metric_keys = [k for k in results[0].keys() if k not in {
            "distortion", "level", "checkpoint", "run_dir", "split_file"
        }]
        for key in metric_keys:
            values = [row[key] for row in results]
            plt.figure()
            plt.plot(levels_sorted, values, marker="o", label="Sensitivity")
            if key in base_metrics:
                plt.axhline(base_metrics[key], color="red", linestyle="--", label="Test baseline")
            plt.xlabel("Distortion level")
            plt.ylabel(key)
            plt.title(f"{args.distortion} - {key}")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            out_path = plot_dir / f"{args.distortion}_{checkpoint_label}_{key}.png"
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()
