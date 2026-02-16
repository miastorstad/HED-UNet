from pathlib import Path
from osgeo import gdal 

DATASET_ROOT = Path("dataset_root")
INPUT_DIR = DATASET_ROOT / "new_test_scenes"

OUTPUT_RESOLUTIONS = {
    "40cm": 0.4,
    "60cm": 0.6,
    "80cm": 0.8,
    "1m": 1,
}

RESAMPLE_ALG = gdal.GRA_Average
OVERWRITE = False

gdal.UseExceptions()

creation_options = [
    "TILED=YES",
    "COMPRESS=DEFLATE",
    "PREDICTOR=2",
    "BIGTIFF=IF_SAFER",
]

tif_files = sorted(
    list(INPUT_DIR.glob("*.tif")) + list(INPUT_DIR.glob("*.tiff"))
)

if not tif_files:
    raise RuntimeError(f"No GeoTIFFs found in {INPUT_DIR}")

print(f"Found {len(tif_files)} orthophotos")

for src_path in tif_files:
    src_ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    if src_ds is None:
        print(f"SKIP (cannot open): {src_path}")
        continue

    stem = src_path.stem

    for label, pixel_size in OUTPUT_RESOLUTIONS.items():
        out_dir = DATASET_ROOT / f"tirol_2022_2023{label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{stem}_{label}.tif"

        if out_path.exists() and not OVERWRITE:
            print(f"SKIP (exists): {out_path}")
            continue

        print(f"Resampling {src_path.name} -> {out_path.name} ({pixel_size} m)")

        out_ds = gdal.Warp(
            destNameOrDestDS=str(out_path),
            srcDSOrSrcDSTab=src_ds,
            format="GTiff",
            xRes=pixel_size,
            yRes=pixel_size,
            resampleAlg=RESAMPLE_ALG,
            targetAlignedPixels=True,
            multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            creationOptions=creation_options,
        )

        if out_ds is None:
            raise RuntimeError(f"GDAL Warp failed for {out_path}")
        
        out_ds = None

    src_ds = None

print("All resampling finished")
