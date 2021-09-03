import logging
import concurrent.futures
import itertools
from typing import Optional, Union

import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import Blosc
from skimage.measure import regionprops_table


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class Image:
    def __init__(self, path: str):
        self.path = path
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


def cut_cells(
    img: np.ndarray,
    cell_data: pd.DataFrame,
    window_size: int,
    cell_stack: Union[zarr.Array, np.ndarray],
    mask_thumbnails: Optional[Union[zarr.Array, np.ndarray]] = None,
    create_mask_thumbnails: bool = False,
) -> None:
    "Cut cells from a given image and write them into the given Zarr or Numpy array."
    for i, c in enumerate(cell_data.itertuples()):
        centroids = np.array([c.Y_centroid, c.X_centroid]).astype(int)
        thumbnail = img[
            centroids[0] : centroids[0] + window_size,
            centroids[1] : centroids[1] + window_size,
        ]
        if create_mask_thumbnails:
            thumbnail = thumbnail == c.CellID
        cell_stack[i, :, :] = thumbnail
        if mask_thumbnails is not None:
            cell_stack[i, :, :] *= mask_thumbnails[i]


def cut_cells_chunked(
    img: np.ndarray,
    cell_data: pd.DataFrame,
    window_size: int,
    cell_stack: zarr.Array,
    dtype: np.dtype,
    mask_thumbnails: Optional[zarr.Array] = None,
    create_mask_thumbnails: bool = False,
    channel_index: Optional[int] = None,
) -> None:
    "Cut cells from a given image, in chunks aligned with the output array, and write them into the given Zarr or Numpy array."
    cell_chunk_size = (
        cell_stack.chunks[1] if len(cell_stack.chunks) == 4 else cell_stack.chunks[0]
    )
    n_cells = len(cell_data)
    window_size_h = window_size // 2
    img = np.pad(img, ((window_size_h, window_size_h), (window_size_h, window_size_h)))
    # Iterate over slices of the data equal to the chunk size in the cell dimension
    for i, (s, e) in enumerate(
        pairwise(itertools.chain(range(0, n_cells, cell_chunk_size), [n_cells]))
    ):
        if i % 1000 == 0:
            logging.info(f"Processed {s} cells")
        # logging.debug(f"Processing chunk {i} with cells {s} - {e}")
        cell_data_c = cell_data.iloc[s:e]
        mask_thumbnails_c = None
        if mask_thumbnails is not None:
            mask_thumbnails_c = mask_thumbnails[s:e, ...]
        cell_stack_c = np.empty((e - s, window_size, window_size), dtype=dtype)
        cut_cells(
            img,
            cell_data_c,
            window_size,
            cell_stack_c,
            mask_thumbnails=mask_thumbnails_c,
            create_mask_thumbnails=create_mask_thumbnails,
        )
        if channel_index is not None:
            cell_stack[channel_index, s:e, ...] = cell_stack_c
        else:
            cell_stack[s:e, ...] = cell_stack_c


def cut_cells_mp(
    image: str,
    cell_data: pd.DataFrame,
    channel_idx: int,
    window_size: int,
    cut_array: zarr.Array,
    mask_thumbnails: Optional[zarr.Array] = None,
) -> None:
    "Load single channel from the given TIFF file and cut out cells."
    logging.info(f"Loading channel {channel_idx}")
    img = Image(image).get_channel(channel_idx)
    cut_cells_chunked(
        img,
        cell_data,
        window_size,
        cut_array,
        dtype=img.dtype,
        mask_thumbnails=mask_thumbnails,
        channel_index=channel_idx,
    )


def find_bbox_size(segmentation_mask: np.ndarray) -> int:
    "Find maximum width or height across all features."
    props = regionprops_table(segmentation_mask, properties=["bbox"])
    return np.max(
        [props["bbox-2"] - props["bbox-0"], props["bbox-3"] - props["bbox-1"]]
    )


def find_chunk_size(shape, sizeof: int, target_size: int = 32 * 1024 * 1024):
    """Given a array of shape [#channels, #cells, height, width] find chunk pattern of the form [1, x, height, width]
    resulting in chunks of the given size.
    """
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
    target_chunk_size: int = 32 * 1024 * 1024,
) -> None:
    "Given an image, segmentation mask, and cell positions, cut out cells and write a stack of cell thumbnails in Zarr format."
    logging.info("Loading segmentation mask")
    segmentation_mask_img = segmentation_mask.get_channel(0)
    # Check if all cell IDs present in the CSV file are also represented in the segmentation mask
    logging.info("Check consistency of cell IDs")
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
    array_chunks = find_chunk_size(
        array_shape, np.dtype(array_dtype).itemsize, target_size=target_chunk_size
    )
    logging.info(f"Using chunks of shape {array_chunks}")
    file = zarr.open(
        destination,
        mode="w",
        shape=array_shape,
        dtype=array_dtype,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        chunks=array_chunks,
    )
    mask_thumbnails = None
    if mask_cells:
        logging.info("Cutting cell mask thumbnails")
        mask_thumbnails = zarr.open(
            f"{destination}_thumbnails",
            mode="w",
            shape=(cell_data.shape[0], window_size, window_size),
            dtype=np.bool_,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            chunks=(array_chunks[1], array_chunks[2], array_chunks[3]),
        )
        cut_cells_chunked(
            segmentation_mask_img,
            cell_data,
            window_size,
            mask_thumbnails,
            dtype=np.bool_,
            create_mask_thumbnails=True,
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        futures = {
            executor.submit(
                cut_cells_mp,
                img.path,
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
                future.result()
                logging.info(f"Channel {i} done")
            except Exception as ex:
                logging.error(f"Future {i} generated an exception: {ex}")
