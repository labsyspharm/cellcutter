import logging
import concurrent.futures
from operator import inv
import pathlib
from typing import Optional, Union, Iterable, Tuple
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from skimage.measure import regionprops_table

from .utils import padded_subset, range_all, pairwise, zip_dir, Image


def cut_cells(
    img: Union[zarr.Array, np.ndarray],
    cell_data: pd.DataFrame,
    window_size: int,
    cell_stack: Union[zarr.Array, np.ndarray],
    mask_thumbnails: Optional[Union[zarr.Array, np.ndarray]] = None,
    create_mask_thumbnails: bool = False,
    channels: Optional[Iterable[int]] = None,
) -> None:
    "Cut cells from a given image and write them into the given Zarr or Numpy array."
    for i, c in enumerate(cell_data.itertuples()):
        centroids = np.array([c.Y_centroid, c.X_centroid]).astype(int)
        thumbnail = padded_subset(
            img, centroids[0], centroids[1], window_size=(window_size, window_size)
        )
        if create_mask_thumbnails:
            thumbnail = thumbnail == c.CellID
        if channels is not None:
            thumbnail = thumbnail[channels, ...]
        cell_stack[..., i, :, :] = thumbnail
        if mask_thumbnails is not None:
            cell_stack[..., i, :, :] *= mask_thumbnails[i]


def cut_cell_range(
    image: str,
    cell_data: pd.DataFrame,
    cell_range: Tuple[int, int],
    window_size: int,
    cut_array: zarr.Array,
    mask_thumbnails: Optional[zarr.Array] = None,
    cache_size: int = 1024 * 1024 * 1024,
    channels: Optional[Iterable[int]] = None,
) -> None:
    "Cut the given range of cells from a TIFF image and write them into the given Zarr or Numpy array."
    image = Image(image, cache_size=cache_size)
    cell_data_subset = cell_data.iloc[cell_range[0] : cell_range[1]]
    cell_stack_temp = np.empty(
        (
            image.n_channels if channels is None else len(channels),
            cell_range[1] - cell_range[0],
            window_size,
            window_size,
        ),
        dtype=image.dtype,
    )
    cut_cells(
        image.zarr,
        cell_data_subset,
        window_size,
        cell_stack_temp,
        mask_thumbnails,
        channels=channels,
    )
    cut_array[..., cell_range[0] : cell_range[1], :, :] = cell_stack_temp


def cut_cell_range_shared_mem(
    image: str,
    address: str,
    cell_data: pd.DataFrame,
    cell_range: Tuple[int, int],
    window_size: int,
    cut_array: zarr.Array,
    mask_thumbnails: Optional[zarr.Array] = None,
    channels: Optional[Iterable[int]] = None,
) -> None:
    "Cut the given range of cells from a TIFF image in shared memory and write them into the given Zarr or Numpy array."
    img = Image(image)
    sm = SharedMemory(name=address)
    img_shape = list(img.base_series.shape)
    if channels is not None:
        img_shape[0] = len(channels)
    img_array = np.ndarray(img_shape, dtype=img.dtype, buffer=sm.buf)
    cell_data_subset = cell_data.iloc[cell_range[0] : cell_range[1]]
    cell_stack_temp = np.empty(
        (
            img.n_channels if channels is None else len(channels),
            cell_range[1] - cell_range[0],
            window_size,
            window_size,
        ),
        dtype=img.dtype,
    )
    cut_cells(
        img_array,
        cell_data_subset,
        window_size,
        cell_stack_temp,
        mask_thumbnails,
        # Input image already only includes the channels we want
        channels=None,
    )
    cut_array[..., cell_range[0] : cell_range[1], :, :] = cell_stack_temp


def find_bbox_size(segmentation_mask: np.ndarray) -> int:
    "Find maximum width or height across all features."
    props = regionprops_table(segmentation_mask, properties=["bbox"])
    return np.max(
        [props["bbox-2"] - props["bbox-0"], props["bbox-3"] - props["bbox-1"]]
    )


def find_chunk_size(
    shape: Tuple[int], sizeof: int, target_bytes: int = 32 * 1024 * 1024
) -> Tuple[int]:
    """Given a array of shape [#channels, #cells, height, width] find chunk pattern of the form [n_channels, x, height, width]
    resulting in chunks of the given size.
    """
    cells_per_chunk = target_bytes // (shape[0] * shape[2] * shape[3] * sizeof)
    return tuple(int(x) for x in (shape[0], cells_per_chunk, shape[2], shape[3]))


def process_all_channels(
    img: Image,
    segmentation_mask: Image,
    cell_data: pd.DataFrame,
    destination: Union[pathlib.Path, str],
    window_size: Optional[int],
    mask_cells: bool = True,
    processes: int = 1,
    target_chunk_size: int = 32 * 1024 * 1024,
    cells_per_chunk: Optional[int] = None,
    channels: Optional[Iterable[int]] = None,
    use_zip: bool = False,
    cache_size: int = 1024 * 1024 * 1024,
) -> None:
    "Given an image, segmentation mask, and cell positions, cut out cells and write a stack of cell thumbnails in Zarr format."
    destination = pathlib.Path(destination)
    if use_zip:
        destination = destination.with_suffix(".zip")
    logging.info("Loading segmentation mask")
    segmentation_mask_img = segmentation_mask.get_channel(0)
    # Check if all cell IDs present in the CSV file are also represented in the segmentation mask
    logging.info("Check consistency of cell IDs")
    cell_ids_not_in_segmentation_mask = np.in1d(cell_data["CellID"], segmentation_mask_img, invert=True)
    n_not_in_segmentation_mask = np.sum(cell_ids_not_in_segmentation_mask)
    del cell_ids_not_in_segmentation_mask
    if n_not_in_segmentation_mask > 0:
        raise ValueError(
            f"{n_not_in_segmentation_mask} cell IDs in the CELL_DATA CSV file are not present in the segmentation mask."
        )
    logging.info("Remove cells from segmentation mask that are not present in the CSV")
    segmentation_mask_img[np.isin(segmentation_mask_img, cell_data["CellID"], invert=True)] = 0
    if window_size is None:
        logging.info("Finding window size")
        window_size = find_bbox_size(segmentation_mask_img)
    logging.info(f"Use window size {window_size}")
    if channels is None:
        channels = np.arange(img.n_channels)
    else:
        channels = np.unique(np.array(channels))
    if channels[0] < 0 or channels[-1] >= img.n_channels:
        raise ValueError(f"Channel indices must be between 0 and {img.n_channels - 1}.")
    array_shape = (len(channels), cell_data.shape[0], window_size, window_size)
    if cells_per_chunk is None:
        array_chunks = find_chunk_size(
            array_shape, np.dtype(img.dtype).itemsize, target_bytes=target_chunk_size
        )
    else:
        array_chunks = tuple(
            int(x) for x in (len(channels), cells_per_chunk, window_size, window_size)
        )
    logging.info(f"Using chunks of shape {array_chunks}")
    # If writing to a zip file, create a temporary directory to store the zarr files
    # and compress them into the zip file at the end. Solves issues with concurrent
    # access to the same zip file.
    store = zarr.DirectoryStore(str(destination)) if not use_zip else zarr.TempStore()
    logging.info(f"Writing thumbnails to {store.path}")
    # Chosing low compression level for speed. Size difference is negligible.
    file = zarr.create(
        store=store,
        overwrite=True,
        shape=array_shape,
        dtype=img.dtype,
        compressor=Blosc(cname="zstd", clevel=2, shuffle=Blosc.SHUFFLE),
        chunks=array_chunks,
    )
    mask_thumbnails = None
    if mask_cells:
        logging.info("Cutting cell mask thumbnails")
        destination_mask = destination.with_stem(f"{destination.stem}_mask")
        mask_store = (
            zarr.DirectoryStore(destination_mask) if not use_zip else zarr.TempStore()
        )
        logging.debug(f"Writing mask thumbnails to {mask_store.path}")
        mask_thumbnails = zarr.create(
            store=mask_store,
            overwrite=True,
            shape=(cell_data.shape[0], window_size, window_size),
            dtype=np.bool_,
            compressor=Blosc(cname="zstd", clevel=2, shuffle=Blosc.SHUFFLE),
            chunks=(array_chunks[1], array_chunks[2], array_chunks[3]),
        )
        mask_thumbnails_temp = np.empty(
            (cell_data.shape[0], window_size, window_size), dtype=np.bool_,
        )
        cut_cells(
            segmentation_mask_img,
            cell_data,
            window_size,
            mask_thumbnails_temp,
            create_mask_thumbnails=True,
        )
        mask_thumbnails[...] = mask_thumbnails_temp[...]
        # If writing to zip files was requested zipping up the directory now
        if use_zip:
            logging.debug(f"Zipping up mask to {destination_mask}")
            zip_dir(
                mask_store.path, destination_mask,
            )
    n_cells = array_shape[1]
    # Only load required channels
    img_shape = (len(channels),) + img.base_series.shape[1:]
    n_bytes_img = img.dtype.itemsize * np.prod(img_shape)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=processes
    ) as executor, SharedMemoryManager() as smm:
        if n_bytes_img < cache_size:
            logging.info(
                f"Image size ({round(n_bytes_img / 1024**2)} MB) is smaller "
                f"than cache size ({round(cache_size / 1024**2)} MB). "
                "Loading entire image into memory."
            )
            raw_sm = smm.SharedMemory(size=n_bytes_img)
            img_array = np.ndarray(img_shape, dtype=img.dtype, buffer=raw_sm.buf)
            img_array[...] = img.zarr.oindex[channels, :, :]
            futures = {
                executor.submit(
                    cut_cell_range_shared_mem,
                    img.path,
                    raw_sm.name,
                    cell_data=cell_data,
                    cell_range=cell_range,
                    window_size=window_size,
                    cut_array=file,
                    mask_thumbnails=mask_thumbnails,
                    channels=channels,
                ): cell_range
                for cell_range in pairwise(range_all(0, n_cells, array_chunks[1]))
            }
        else:
            logging.warn(
                f"Image size ({round(n_bytes_img / 1024**2)} MB) is larger than "
                f"cache size ({round(cache_size / 1024**2)} MB). "
                "This results in many reads from disk and will be slow. Consider "
                "increasing the cache size."
            )
            futures = {
                executor.submit(
                    cut_cell_range,
                    img.path,
                    cell_data=cell_data,
                    cell_range=cell_range,
                    window_size=window_size,
                    cut_array=file,
                    mask_thumbnails=mask_thumbnails,
                    cache_size=cache_size // processes,
                    channels=channels,
                ): cell_range
                for cell_range in pairwise(range_all(0, n_cells, array_chunks[1]))
            }
        for future in concurrent.futures.as_completed(futures):
            cell_range = futures[future]
            try:
                future.result()
                logging.info(f"Cells {cell_range[0]}-{cell_range[1]} done")
            except Exception as ex:
                logging.error(
                    f"Error processing cells {cell_range[0]}-{cell_range[1]}: {ex}"
                )
                raise ex
    if use_zip:
        logging.info("Zipping up thumbnails")
        logging.debug(f"Zipping up to {destination}")
        zip_dir(store.path, destination)
