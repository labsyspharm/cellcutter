import itertools
import pathlib
import zipfile
from multiprocessing.shared_memory import SharedMemory
from typing import Union, Tuple, Any

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
    img: Union[zarr.Array, np.ndarray], x_start: int, x_stop: int, y_start: int, y_stop: int, fill_value: Any = 0
) -> np.ndarray:
    """Return a slice of the array with the given window size padded with fill_value if the slice is out of bounds.
    Assumes that the coordinates are the last two dimensions of the array.
    """
    output_shape = img.shape[:-2] + (
        y_stop - y_start,
        x_stop - x_start
    )
    output_array = np.full(output_shape, fill_value, dtype=img.dtype)

    row_start = max(y_start, 0)
    row_end = min(y_stop, img.shape[-2])
    col_start = max(x_start, 0)
    col_end = min(x_stop, img.shape[-1])

    output_array[
        ...,
        row_start - y_start:row_end - y_start,
        col_start - x_start:col_end - x_start
    ] = img[..., row_start:row_end, col_start:col_end]

    return output_array


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
