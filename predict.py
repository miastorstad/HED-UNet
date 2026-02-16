#!/usr/bin/env python3
"""
Run inference on new orthophotos and export binary crevasse maps as GeoTIFFs.

Example:
    python predict.py --run-dir logs/2026-01-20_13-40-33 \
        --input-dir /path/to/new/images --output-dir /path/to/output
"""
import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio as rio
import yaml

from deep_learning import get_model


def parse_best_f1_threshold(metrics_path):
    if not metrics_path.exists():
        return 0.5
    best_f1 = -1.0
    best_threshold = 0.5
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+-\s+Val:.*SegF1:\s+([0-9.]+).*SegThreshold:\s+([0-9.]+)"
    )
    with metrics_path.open("r") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            f1 = float(match.group(2))
            threshold = float(match.group(3))
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    return best_threshold


def load_stats(stats_path):
    stats = np.load(stats_path)
    p2 = torch.from_numpy(stats["p2"].astype("float32")).view(-1, 1, 1)
    p98 = torch.from_numpy(stats["p98"].astype("float32")).view(-1, 1, 1)
    return p2, p98


def compute_stats_from_dir(input_dir, stats_path, max_samples_per_image):
    rng = np.random.default_rng(0)
    samples = []
    for tif in sorted(Path(input_dir).glob("*.tif")):
        with rio.open(tif) as src:
            arr = src.read()  # C, H, W
        c, _, _ = arr.shape
        flat = arr.reshape(c, -1)
        n = flat.shape[1]
        if n > max_samples_per_image:
            idx = rng.choice(n, size=max_samples_per_image, replace=False)
            flat = flat[:, idx]
        samples.append(flat)
    if not samples:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")
    all_pixels = np.concatenate(samples, axis=1)
    p2 = np.percentile(all_pixels, 2, axis=1)
    p98 = np.percentile(all_pixels, 98, axis=1)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, p2=p2, p98=p98)
    print(f"Saved stats to {stats_path}")


def normalize_tile(tile, p2, p98, eps=1e-7):
    tile = tile.to(torch.float32)
    if tile.shape[0] > 3:
        tile = tile[:3]
    p2 = p2[: tile.shape[0]].to(tile.device)
    p98 = p98[: tile.shape[0]].to(tile.device)
    tile = torch.clamp(tile, min=p2, max=p98)
    denom = (p98 - p2) + eps
    tile = (tile - p2) / denom
    return tile


def iter_tiles(height, width, tile_size, stride):
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            yield y, x


def run_inference_on_image(
    model,
    input_path,
    output_path,
    prob_output_path,
    p2,
    p98,
    threshold,
    tile_size,
    stride,
    batch_size,
    device,
):
    with rio.open(input_path) as src:
        profile = src.profile
        height = src.height
        width = src.width

        acc = np.zeros((height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        tiles = []
        positions = []
        for y, x in iter_tiles(height, width, tile_size, stride):
            window = rio.windows.Window(x, y, tile_size, tile_size)
            data = src.read(window=window).astype("float32")
            tile = torch.from_numpy(data)
            tile = normalize_tile(tile, p2, p98)
            tiles.append(tile)
            positions.append((y, x))

            if len(tiles) >= batch_size:
                _flush_tiles(model, tiles, positions, acc, count, threshold, device)
                tiles.clear()
                positions.clear()

        if tiles:
            _flush_tiles(model, tiles, positions, acc, count, threshold, device)

    avg = np.zeros_like(acc)
    valid = count > 0
    avg[valid] = acc[valid] / count[valid]
    mask = (avg >= threshold).astype(np.uint8)

    profile.update(
        dtype=rio.uint8,
        count=1,
        compress="lzw",
        photometric="MINISBLACK",
        nodata=255,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        # Remove partially written files from prior failed runs.
        output_path.unlink()
    with rio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)

    if prob_output_path is not None:
        prob_profile = profile.copy()
        prob_profile.update(dtype=rio.float32, nodata=None)
        if "nodata" in prob_profile and prob_profile["nodata"] is None:
            del prob_profile["nodata"]
        prob_output_path.parent.mkdir(parents=True, exist_ok=True)
        if prob_output_path.exists():
            prob_output_path.unlink()
        with rio.open(prob_output_path, "w", **prob_profile) as dst:
            dst.write(avg.astype(np.float32), 1)


@torch.no_grad()
def _flush_tiles(model, tiles, positions, acc, count, threshold, device):
    batch = torch.stack(tiles).to(device)
    preds = model(batch)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    seg_prob = torch.sigmoid(preds[:, 1]).detach().cpu().numpy()
    for idx, (y, x) in enumerate(positions):
        acc[y:y + seg_prob.shape[1], x:x + seg_prob.shape[2]] += seg_prob[idx]
        count[y:y + seg_prob.shape[1], x:x + seg_prob.shape[2]] += 1.0


def main():
    parser = argparse.ArgumentParser(description="Predict crevasse maps on new orthophotos.")
    parser.add_argument("--run-dir", required=True, help="Path to run directory with config and checkpoints.")
    parser.add_argument("--checkpoint", default="best_f1",
                        help="Epoch number, 'best_f1', or a .pt path.")
    parser.add_argument("--input-dir", required=True, help="Folder with GeoTIFFs to process.")
    parser.add_argument("--output-dir", required=True, help="Folder to write output masks.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override segmentation threshold (defaults to best F1 threshold).")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size (pixels).")
    parser.add_argument("--stride", type=int, default=100, help="Stride between tiles (pixels).")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--stats", default="dataset_root/crevasse_stats.npz",
                        help="Path to normalization stats npz.")
    parser.add_argument("--compute-stats", action="store_true",
                        help="Compute stats from input-dir and save to --stats.")
    parser.add_argument("--stats-samples-per-image", type=int, default=2_000_000,
                        help="Max pixels sampled per image when computing stats.")
    parser.add_argument("--prob-output-dir", default=None,
                        help="Optional folder to write float32 probability maps.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yml in {run_dir}")
    config = yaml.load(config_path.open(), Loader=yaml.SafeLoader)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        if args.checkpoint == "best_f1":
            checkpoint = run_dir / "checkpoints" / "best_f1.pt"
        elif args.checkpoint.isdigit():
            epoch = int(args.checkpoint)
            checkpoint = run_dir / "checkpoints" / f"{epoch:02d}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    threshold = args.threshold
    if threshold is None:
        threshold = parse_best_f1_threshold(run_dir / "metrics.txt")

    modelclass = get_model(config["model"])
    model = modelclass(**config["model_args"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    prob_output_dir = Path(args.prob_output_dir) if args.prob_output_dir else None
    stats_path = Path(args.stats)
    if args.compute_stats or not stats_path.exists():
        compute_stats_from_dir(input_dir, stats_path, args.stats_samples_per_image)
    p2, p98 = load_stats(stats_path)
    tifs = sorted(input_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")

    for tif in tifs:
        out_path = output_dir / f"{tif.stem}_crevasse.tif"
        prob_path = None
        if prob_output_dir is not None:
            prob_path = prob_output_dir / f"{tif.stem}_prob.tif"
        run_inference_on_image(
            model,
            tif,
            out_path,
            prob_path,
            p2,
            p98,
            threshold,
            args.tile_size,
            args.stride,
            args.batch_size,
            device,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
