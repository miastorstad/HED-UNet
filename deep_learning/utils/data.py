from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio as rio
import shutil

TILESIZE = 512

class InriaDataset(Dataset):
    """
    Random access Dataset for datasets of tiffs stored like this:
        images/<tile1>.tif
        images/<tile2>.tif
        ...
        gt/<tile1>.tif
        gt/<tile2>.tif
        ...
    """
    def __init__(self, root, parts=['images', 'gt'], transform=None, suffix='.tif'):
        self.root = Path(root)
        self.parts = parts

        first = self.root / parts[0]
        filenames = list(sorted([x.name for x in first.glob('*' + suffix)]))
        self.index = [[self.root / p / x for p in parts] for x in filenames]
        self.transform = transform
        self.cache_dir = self.root / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.build_tile_cache()

    def build_tile_cache(self):
        self.tiles = []
        for files in tqdm(self.index, desc='Building/Checking Tile Cache'):
            file = files[0]
            cache_dir = self.cache_dir / file.stem
            lockfile = cache_dir / 'lock.file'
            if not cache_dir.is_dir() or lockfile.exists():
                if lockfile.exists():
                    shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=False)
                lockfile.touch()

                data_out = {}
                for file in files:
                    with rio.open(file) as raster:
                        data = raster.read()
                        _, H, W = data.shape
                        if file.parent.name == 'gt':
                            assert len(np.unique(data)) <= 2, f'{file}: {np.unique(data)}'
                            data = data > 0
                        data_out[file.parent.name] = data

                # Baumhoer et al. (2025): 512×512 tiles, stride 100 pixels
                for y in range(0, H - TILESIZE + 1, 100):
                    for x in range(0, W - TILESIZE + 1, 100):
                        tile = {
                            p: data_out[p][:, y:y + TILESIZE, x:x + TILESIZE]
                            for p in self.parts
                    }
                    np.savez(cache_dir / f'{y:04d}_{x:04d}.npz', **tile)

                lockfile.unlink()
            self.tiles += list(sorted(cache_dir.glob('*.npz')))


    def __getitem__(self, idx):
        file = self.tiles[idx]
        data_dict = np.load(file)

        data = []
        for p in self.parts:
            arr = data_dict[p]
            # Torch doesn't support uint16, so cast to float32
            if arr.dtype == np.uint16:
                arr = arr.astype(np.float32)
            data.append(torch.from_numpy(arr))

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.tiles)


class PTDataset(Dataset):
    """
    Random access Dataset for datasets of pytorch tensors stored like this:
        data/images/1.pt
        data/images/2.pt
        ...
        data/masks/1.pt
        data/masks/2.pt
        ...
    """
    def __init__(self, root, parts, transform=None, suffix='.pt'):
        self.root = Path(root)
        self.parts = parts

        first = self.root / parts[0]
        filenames = list(sorted([x.name for x in first.glob('*' + suffix)]))
        self.index = [[self.root / p / x for p in parts] for x in filenames]
        self.transform = transform

    def __getitem__(self, idx):
        files = self.index[idx]
        data = [torch.load(f) for f in files]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.index)


class Augment(Dataset):
    def __init__(self, dataset, brightness=0.0, brightness_range=None):
        self.dataset = dataset
        self.len = 8 * len(self.dataset)
        if brightness_range is not None:
            if len(brightness_range) != 2:
                raise ValueError("brightness_range must be a (min, max) pair")
            self.brightness_range = tuple(brightness_range)
        elif brightness and brightness > 0:
            self.brightness_range = (max(0.0, 1.0 - brightness), 1.0 + brightness)
        else:
            self.brightness_range = None

    def __getitem__(self, idx):
        idx, carry = divmod(idx, 8)
        carry, flipx = divmod(carry, 2)
        transpose, flipy = divmod(carry, 2)

        diry = 2 * flipy - 1
        dirx = 2 * flipx - 1
        base = self.dataset[idx]
        augmented = []
        for field in base:
            field = field.numpy()
            field = field[:, ::diry, ::dirx]
            if transpose == 1:
                field = field.transpose(0, 2, 1)
            augmented.append(torch.from_numpy(field.copy()))

        if self.brightness_range is not None:
            low, high = self.brightness_range
            if high < low:
                low, high = high, low
            factor = torch.empty(1).uniform_(low, high).item()
            img = augmented[0].to(torch.float32) * factor
            augmented[0] = torch.clamp(img, 0.0, 1.0)

        return tuple(augmented)

    def __len__(self):
        return len(self.dataset) * 8
