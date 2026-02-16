# HED-UNet for crevasse detection
A part of my masters thesis on crevasse detection using deep learning. A HED-UNet model and training pipeline for crevasse detection using orthophotos. This repo includes training, evaluation, and tiled inference for GeoTIFF imagery. 


Original code from https://github.com/khdlr/HED-UNet

## Overview 
- Architecture: HED-UNet with deep supervision and edge head.
- Task: binary crevasse segmentation + edge map prediction
- Outputs: segmentation probability maps and binary masks.

## Repository structure 
- train.py         -> training entrypoint
- predict.py       -> tiled inference on new GeoTIFF
- config.yml       -> training configuration
- data_loading.py  -> dataset setup and normalization
- dataset_root/    -> expected dataset root (not committed)

## Data 
Expected layout (GeoTIFFs, matched by filename):

dataset_root/
images/
tile_0001.tif
tile_0002.tif
gt/ 
tile_0001.tif
tile_0002.tif

- Ground truth masks should be binary (0/1).
- A tile cache is built under dataset_root/cache/ for 512x512 tiled with stride 100.

### Normalization stats 
Training expects dataset_root/crevasse_stats.npz with 2nd/98th percentiles per band. 
To compute: 
python compute_stats.py

## Training
python train.py --config config.yml 

Outputs go to logs/<timestamp>/ :
- config.yml (copied)
- checkpoints/
- metrics.txt
- tensorboard/
- figures/ (qualitatice previews)

## Inference 
python predict.py --run-dir logs/<run>
--input-dir /path/to/orthos
--output-dir /path/to/masks

Useful flags:
- --checkpoint best_f1|<epoch>|/path/to.ckpt
- --threshold 0.5 (override best f1 theshold)
- --compute-stats (compute stats from new imagery)
- --prob-output-dir /path/to/prob_maps

## Configuration 
Key knobs in config.yml:
- model / model_args.*                                           -> selects the architecture and its capacity.
  - input_channels                                               -> should match your imagery bands (this code uses the first 3).
  - output_channels                                              -> is 2 (seg + edge).
  - base_channels, stack_height, merging, and deep_supervision   -> control network depth and skip-connection behavior.
- feature_pyramid                                                -> enables multi-scale feature fusion before the output heads.
- loss_args / loss_weights
  - loss_args.type                                               -> chooses the loss (e.g. FocalLoss).
  - loss_weights.seg and .edge                                   -> weight segmentation vs. edge supervision.
- batch_size, learning_rate, epochs                              -> core training hyperparameters.
- early_stopping, patience, early_stopping_tolerance             -> controls early stopping if validation improvment stalls.
- augmentation                                                   -> brightness jitter for RGB normalization. Use brightnedd or brightness_range to expand/limit intensity variation.
- balances_sampling, balanced_sampling_*                         -> If enabled, tiles are sampled with weigths based on positive-pixel fraction to reduce class imbalance.
  - power                                                        -> sets how aggressively rare positived are upweighted.
  - max_weight                                                   -> caps weights.
- data_threads, split_seed, run_test                             -> data loader workers, train/val split seed, and whether to run the test pass after training.
- visualization_tiles                                            -> tile indices used for consistent qualitative snapshots during training. 
