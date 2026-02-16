# compute_stats.py  (run once)
import numpy as np
from pathlib import Path
import rasterio as rio

root = Path("dataset_root/images")  

all_pixels = []

for tif in root.glob("*.tif"):
    with rio.open(tif) as ds:
        arr = ds.read()  # C,H,W
        c, h, w = arr.shape
        # sample a subset if huge:
        arr = arr.reshape(c, -1)
        all_pixels.append(arr)

all_pixels = np.concatenate(all_pixels, axis=1)  # C, N

p2 = np.percentile(all_pixels, 2, axis=1)
p98 = np.percentile(all_pixels, 98, axis=1)

np.savez("dataset_root/crevasse_stats.npz", p2=p2, p98=p98)
print("Saved stats to dataset_root/crevasse_stats.npz")
