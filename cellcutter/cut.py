import logging
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import Blosc
from skimage.measure import regionprops_table


class Image:
    def __init__(self, path: str):
        self.image = tifffile.TiffFile(path)
        self.base_series = self.image.series[0]

    def get_channel(self, channel_index: int) -> np.ndarray:
        return self.base_series.pages[channel_index].asarray()

    @property
    def n_channels(self) -> int:
        if (len(self.base_series.shape) == 2):
            return 1
        else:
            return self.base_series.shape[0]


# Efficient way to copy crops from an image where coordinates
# can be beyond the image boundaries
# Avoids copying the whole image with padding
# https://stackoverflow.com/a/55391329/4603385
def fill_crop(img: np.ndarray, pos: int, crop: np.ndarray) -> None:
    img_shape, pos, crop_shape = (
        np.array(img.shape),
        np.array(pos),
        np.array(crop.shape),
    )
    end = pos + crop_shape
    # Calculate crop slice positions
    crop_low = np.clip(0 - pos, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
    crop_slices = tuple(slice(low, high) for low, high in zip(crop_low, crop_high))
    # Calculate img slice positions
    pos = np.clip(pos, a_min=0, a_max=img_shape)
    end = np.clip(end, a_min=0, a_max=img_shape)
    img_slices = tuple(slice(low, high) for low, high in zip(pos, end))
    crop[crop_slices] = img[img_slices]


# Processes single channel
def cut_cells(
    img: np.ndarray,
    segmentation_mask: np.ndarray,
    cell_data: pd.DataFrame,
    window_size: int,
    mask_cells: bool = True,
    cell_stack: Optional[np.ndarray] = None,
) -> np.ndarray:
    if cell_stack is None:
        cell_stack = np.zeros(
            (cell_data.shape[0], window_size, window_size), dtype=img.dtype
        )
    for i, c in enumerate(cell_data.itertuples()):
        centroids = np.array([c.Y_centroid, c.X_centroid]).astype(int) - window_size // 2
        fill_crop(img, centroids, cell_stack[i, :, :])
        if mask_cells:
            mask = np.zeros((window_size, window_size), dtype=segmentation_mask.dtype)
            fill_crop(segmentation_mask, centroids, mask)
            cell_stack[i, mask != c.CellID] = 0
    return cell_stack


def find_bbox_size(segmentation_mask: np.ndarray) -> int:
    props = regionprops_table(segmentation_mask, properties=["bbox"])
    return np.max(
        [props["bbox-2"] - props["bbox-0"], props["bbox-3"] - props["bbox-1"]]
    )


def save_cells_all_channels(
    img: Image,
    segmentation_mask: Image,
    cell_data: pd.DataFrame,
    destination: str,
    window_size: Optional[int],
    mask_cells: bool = True,
) -> None:
    segmentation_mask_img = segmentation_mask.get_channel(0)
    if window_size is None:
        logging.info("Finding window size")
        window_size = find_bbox_size(segmentation_mask_img)
        logging.info(f"Use window size {window_size}")
    file = zarr.open(
        destination,
        mode="w",
        shape=(img.n_channels, cell_data.shape[0], window_size, window_size),
        dtype=img.get_channel(0).dtype,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    for i in range(img.n_channels):
        logging.info(f"Processing channel {i}")
        channel_img = img.get_channel(i)
        cell_stack = cut_cells(
            channel_img,
            segmentation_mask_img,
            cell_data,
            window_size,
            mask_cells=mask_cells,
        )
        logging.info(f"Writing thumbnails for channel {i}")
        file[i, ...] = cell_stack
