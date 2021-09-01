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
        if len(self.base_series.shape) == 2:
            return 1
        else:
            return self.base_series.shape[0]


# Processes single channel
def cut_cells(
    img: np.ndarray,
    cell_data: pd.DataFrame,
    window_size: int,
    mask_thumbnails: Optional[np.ndarray] = None,
    cell_stack: Optional[np.ndarray] = None,
) -> np.ndarray:
    if cell_stack is None:
        cell_stack = np.empty(
            (cell_data.shape[0], window_size, window_size), dtype=img.dtype
        )
    window_size_h = window_size // 2
    img = np.pad(img, ((window_size_h, window_size_h), (window_size_h, window_size_h)))
    for i, c in enumerate(cell_data.itertuples()):
        centroids = np.array([c.Y_centroid, c.X_centroid]).astype(int)
        cell_stack[i, :, :] = img[
            centroids[0] : centroids[0] + window_size,
            centroids[1] : centroids[1] + window_size,
        ]
        if mask_thumbnails is not None:
            cell_stack[i, :, :] *= mask_thumbnails[i]
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
    mask_thumbnails = None
    if mask_cells:
        window_size_h = window_size // 2
        segmentation_mask_img_padded = np.pad(
            segmentation_mask_img,
            ((window_size_h, window_size_h), (window_size_h, window_size_h)),
        )
        mask_thumbnails = np.empty(
            (cell_data.shape[0], window_size, window_size), dtype=bool
        )
        for i, c in enumerate(cell_data.itertuples()):
            centroids = np.array([c.Y_centroid, c.X_centroid]).astype(int)
            mask_thumbnails[i, :, :] = (
                segmentation_mask_img_padded[
                    centroids[0] : centroids[0] + window_size,
                    centroids[1] : centroids[1] + window_size,
                ]
                == c.CellID
            )
    for i in range(img.n_channels):
        logging.info(f"Processing channel {i}")
        channel_img = img.get_channel(i)
        cell_stack = cut_cells(
            channel_img,
            cell_data,
            window_size,
            mask_thumbnails=mask_thumbnails,
        )
        logging.info(f"Writing thumbnails for channel {i}")
        file[i, ...] = cell_stack
