import logging
import concurrent.futures
import tempfile
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


def cut_cells_mp(
    image: Image,
    cell_data: pd.DataFrame,
    channel_idx: int,
    window_size: int,
    cut_array: zarr.Array,
    mask_thumbnails: Optional[str] = None,
) -> None:
    img = image.get_channel(channel_idx)
    if mask_thumbnails is not None:
        mask_thumbnails = np.load(mask_thumbnails)
    cell_stack = cut_cells(img, cell_data, window_size, mask_thumbnails=mask_thumbnails)
    cut_array[channel_idx, ...] = cell_stack


def find_bbox_size(segmentation_mask: np.ndarray) -> int:
    props = regionprops_table(segmentation_mask, properties=["bbox"])
    return np.max(
        [props["bbox-2"] - props["bbox-0"], props["bbox-3"] - props["bbox-1"]]
    )


def find_chunk_size(shape, sizeof: int, target_size: int = 32 * 1024 * 1024):
    total_size = np.prod(shape) * sizeof
    n_cell_chunks = total_size / (target_size * shape[0])
    return tuple(
        int(x) for x in (1, np.ceil(shape[1] / n_cell_chunks), shape[2], shape[3])
    )


def process_all_channels(
    img: Image,
    segmentation_mask: Image,
    cell_data: pd.DataFrame,
    destination: str,
    window_size: Optional[int],
    mask_cells: bool = True,
    processes: int = 1,
) -> None:
    segmentation_mask_img = segmentation_mask.get_channel(0)
    # Check if all cell IDs present in the CSV file are also represented in the segmentation mask
    cell_ids_in_segmentation_mask = np.unique(segmentation_mask_img)
    n_not_in_segmentation_mask = set(cell_data["CellID"]) - set(
        cell_ids_in_segmentation_mask
    )
    if len(n_not_in_segmentation_mask) > 0:
        raise SystemError(
            f"{len(n_not_in_segmentation_mask)} cell IDs in the CELL_DATA CSV file are not present in the segmentation mask."
        )
    # Remove cells from segmentation mask that are not present in the CSV
    segmentation_mask_img[~np.isin(segmentation_mask_img, cell_data["CellID"])] = 0
    if window_size is None:
        logging.info("Finding window size")
        window_size = find_bbox_size(segmentation_mask_img)
        logging.info(f"Use window size {window_size}")
    array_shape = (img.n_channels, cell_data.shape[0], window_size, window_size)
    array_dtype = img.get_channel(0).dtype
    array_chunks = find_chunk_size(array_shape, np.dtype(array_dtype).itemsize)
    logging.info(f"Using {array_chunks} chunks")
    file = zarr.open(
        destination,
        mode="w",
        shape=array_shape,
        dtype=array_dtype,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        chunks=array_chunks,
    )
    mask_thumbnails = None
    if mask_cells:
        window_size_h = window_size // 2
        segmentation_mask_img_padded = np.pad(
            segmentation_mask_img,
            ((window_size_h, window_size_h), (window_size_h, window_size_h)),
        )
        mask_thumbnails = np.empty(
            (cell_data.shape[0], window_size, window_size), dtype=np.bool_
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
        mask_temp = tempfile.mkstemp(suffix=".npy")
        np.save(mask_temp[1], mask_thumbnails)
        mask_thumbnails = mask_temp[1]
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as executor:
        # Start the load operations and mark each future with its URL
        futures = {
            executor.submit(
                cut_cells_mp,
                img,
                cell_data,
                i,
                window_size,
                file,
                mask_thumbnails=mask_thumbnails,
            ): i
            for i in range(img.n_channels)
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                data = future.result()
                logging.info(f"Future {i} done")
            except Exception as ex:
                logging.error(f"Future {i} generated an exception: {ex}")
