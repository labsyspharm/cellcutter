import itertools
import pathlib
import zipfile
from multiprocessing.shared_memory import SharedMemory
from typing import Union, Tuple, Optional

import numpy as np
import tifffile
import zarr


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def range_all(start, stop, step):
    "Like range() but with stop included"
    for i in range(start, stop, step):
        yield i
    yield stop


def zip_dir(dir: Union[pathlib.Path, str], zip_file: Union[pathlib.Path, str]) -> None:
    "Zip the contents of the given directory into a ZIP file."
    dir = pathlib.Path(dir)
    # Don't compress the ZIP file. Zarr arrays are already compressed.
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in dir.rglob("*"):
            zf.write(f, f.relative_to(dir))


class SharedNumpyArraySpec:
    def __init__(self, address: str, shape: Tuple[int, ...], dtype: np.dtype):
        self.address = address
        self.shape = shape
        self.dtype = dtype


def padded_subset(
    img: Union[zarr.Array, np.ndarray], x: int, y: int, window_size: Tuple[int, int]
) -> np.ndarray:
    "Return a subset of the image with the given window size padded with zero if window is partially outside the image"
    wh = img.shape[-2:]
    s = (
        (x - window_size[0] // 2, x + window_size[0] - window_size[0] // 2),
        (y - window_size[1] // 2, y + window_size[1] - window_size[1] // 2),
    )
    s_img = (
        (max(0, s[0][0]), min(wh[0], s[0][1])),
        (max(0, s[1][0]), min(wh[1], s[1][1])),
    )
    s_out = (
        (
            max(0, -s[0][0]),
            min(window_size[0], window_size[0] - s[0][1] + wh[0]),
        ),
        (
            max(0, -s[1][0]),
            min(window_size[1], window_size[1] - s[1][1] + wh[1]),
        ),
    )
    out = np.zeros(img.shape[:-2] + (window_size[0], window_size[1]), dtype=img.dtype)
    out[..., s_out[0][0] : s_out[0][1], s_out[1][0] : s_out[1][1]] = img[
        ..., s_img[0][0] : s_img[0][1], s_img[1][0] : s_img[1][1]
    ]
    return out


class Image:
    def __init__(self, path: str, cache_size: int = 1024 * 1024 * 1024):
        self.path = path
        self.image = tifffile.TiffFile(path)
        self.base_series = self.image.series[0]
        self.cache_size = cache_size
        self.zarr = zarr.open(
            zarr.LRUStoreCache(self.image.aszarr(series=0), self.cache_size), mode="r"
        )
        # If we get a group back assume that the first group is highest resolution
        if isinstance(self.zarr, zarr.Group):
            self.zarr = self.zarr[0]

    def get_channel(self, channel_index: int) -> np.ndarray:
        return self.base_series.pages[channel_index].asarray()

    @property
    def width_height(self) -> Tuple[int, int]:
        return tuple(self.base_series.shape)[-2:]

    @property
    def n_channels(self) -> int:
        if len(self.base_series.shape) == 2:
            return 1
        else:
            return self.base_series.shape[0]

    @property
    def dtype(self):
        return self.base_series.dtype
