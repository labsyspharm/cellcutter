import logging
import concurrent.futures
import pathlib
import warnings
from typing import Optional, Union, Iterable, Tuple
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from skimage.measure import regionprops_table

from .utils import (
    padded_subset,
    range_all,
    pairwise,
    zip_dir,
    Image,
    SharedNumpyArraySpec,
)


class DuplicateChannelWarning(Warning):
    def __init__(self, *args, **kwargs):
        channels = kwargs.pop("duplicate_channels")
        super().__init__(*args, **kwargs)
        self.duplicate_channels = channels

    def __str__(self):
        return (
            "The given channels are not unique. "
            f"Channels {' '.join(str(x) for x in self.duplicate_channels)} will appear more than once in the output file."
        )


def cut_cells(
    img: Union[zarr.Array, np.ndarray],
    roi_data: pd.DataFrame,
    cell_stack: Union[zarr.Array, np.ndarray],
    mask_thumbnails: Optional[Union[zarr.Array, np.ndarray]] = None,
    create_mask_thumbnails: bool = False,
    channels: Optional[Iterable[int]] = None,
) -> None:
    "Cut cells from a given image and write them into the given Zarr or Numpy array."
    for i, c in enumerate(roi_data.itertuples()):
        thumbnail = padded_subset(
            img, c.X_start, c.X_stop, c.Y_start, c.Y_stop
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
    window_size: Tuple[int, int],
    cut_array: zarr.Array,
    mask_thumbnails: Optional[Union[zarr.Array, np.ndarray]] = None,
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
            window_size[0],
            window_size[1],
        ),
        dtype=image.dtype,
    )
    cut_cells(
        image.zarr,
        cell_data_subset,
        cell_stack_temp,
        mask_thumbnails,
        channels=channels,
    )
    cut_array[..., cell_range[0] : cell_range[1], :, :] = cell_stack_temp
    try:
        logging.info(
            f"Cache hits: {image.zarr.store.hits} Cache misses: {image.zarr.store.misses}"
        )
    except Exception:
        pass


def cut_cell_range_shared_mem(
    img_spec: SharedNumpyArraySpec,
    cell_data: pd.DataFrame,
    cell_range: Tuple[int, int],
    window_size: Tuple[int, int],
    cut_array: zarr.Array,
    mask_thumbnails_spec: Optional[SharedNumpyArraySpec] = None,
) -> None:
    "Cut the given range of cells from a TIFF image in shared memory and write them into the given Zarr or Numpy array."
    try:
        img_sm = SharedMemory(img_spec.address)
        img_array = np.ndarray(
            img_spec.shape,
            dtype=img_spec.dtype,
            buffer=img_sm.buf,
        )
        mask_thumbnails = None
        if mask_thumbnails_spec is not None:
            mask_sm = SharedMemory(mask_thumbnails_spec.address)
            mask_thumbnails = np.ndarray(
                mask_thumbnails_spec.shape, dtype=mask_thumbnails_spec.dtype, buffer=mask_sm.buf
            )
        cell_data_subset = cell_data.iloc[cell_range[0] : cell_range[1]]
        cell_stack_temp = np.empty(
            (
                img_array.shape[0],
                cell_range[1] - cell_range[0],
                window_size[0],
                window_size[1],
            ),
            dtype=img_array.dtype,
        )
        cut_cells(
            img_array,
            cell_data_subset,
            cell_stack_temp,
            mask_thumbnails,
            # Input image already only includes the channels we want
            channels=None,
        )
        cut_array[..., cell_range[0] : cell_range[1], :, :] = cell_stack_temp
    finally:
        img_sm.close()
        if mask_thumbnails_spec is not None:
            mask_sm.close()


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


def rois_from_cell_data(
    cell_data: pd.DataFrame,
    window_size: Optional[Tuple[int, int]] = None,
    segmentation_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Tuple[int, int], Optional[np.ndarray]]:
    if segmentation_mask is not None:
        logging.info(
            "Check if all cell IDs from the CSV are represented in the segmentation mask"
        )
        cell_ids_not_in_segmentation_mask = np.in1d(
            cell_data["CellID"], segmentation_mask, invert=True
        )
        n_not_in_segmentation_mask = np.sum(cell_ids_not_in_segmentation_mask)
        if n_not_in_segmentation_mask > 0:
            raise ValueError(
                f"{n_not_in_segmentation_mask} cell IDs in the CELL_DATA CSV file are not represented in the segmentation mask. "
                "Please check that the segmentation mask contains all cell IDs present in the CSV file. "
                f"The first 10 problematic IDs are {' '.join(str(x) for x in cell_data['CellID'][cell_ids_not_in_segmentation_mask][:10])}"
            )
        del cell_ids_not_in_segmentation_mask
        logging.info("Remove cells from segmentation mask that are not present in the CSV")
        segmentation_mask[
            np.isin(segmentation_mask, cell_data["CellID"], invert=True)
        ] = 0
    if window_size is None:
        logging.info("Finding window size")
        window_size = find_bbox_size(segmentation_mask)
        window_size = (window_size, window_size)
        logging.info(f"Window size automatically set to {window_size}")
    cell_data["X_start"] = cell_data["X_centroid"].astype(int) - window_size[0] // 2
    cell_data["X_stop"] = cell_data["X_start"] + window_size[0]
    cell_data["Y_start"] = cell_data["Y_centroid"].astype(int) - window_size[1] // 2
    cell_data["Y_stop"] = cell_data["Y_start"] + window_size[1]
    return cell_data, window_size, segmentation_mask


def rois_from_grid(
    img: Union[str, Image],
    window_size: Tuple[int, int],
    step_size: int,
    offset: Tuple[int, int] = (0, 0),
) -> pd.DataFrame:
    if not isinstance(img, Image):
        img = Image(img)
    width, height = img.width_height
    x_starts = np.arange(offset[0], width, step_size)
    y_starts = np.arange(offset[1], height, step_size)
    x_i, y_i = (v.ravel() for v in np.meshgrid(x_starts, y_starts))
    tile_df = pd.DataFrame(
        {
            "TileID": np.arange(x_i.size),
            "X_start": x_i,
            "X_stop": x_i + window_size[0],
            "Y_start": y_i,
            "Y_stop": y_i + window_size[1]
        }
    )
    tile_df["OutOfBounds"] = (
        (tile_df["X_stop"] > width)
        | (tile_df["Y_stop"] > height)
        | (tile_df["X_start"] < 0)
        | (tile_df["Y_start"] < 0)
    )
    return tile_df


def process_image(
    img: Union[str, Image],
    roi_data: pd.DataFrame,
    segmentation_mask: Optional[np.ndarray],
    destination: Union[pathlib.Path, str],
    window_size: Optional[Tuple[int, int]] = None,
    mask_cells: bool = True,
    processes: int = 1,
    target_chunk_size: int = 32 * 1024 * 1024,
    cells_per_chunk: Optional[int] = None,
    channels: Optional[Iterable[int]] = None,
    use_zip: bool = False,
    cache_size: int = 1024 * 1024 * 1024,
) -> None:
    "Given an image, cell positions, cut out cells and write a stack of cell thumbnails in Zarr format."
    if not isinstance(img, Image):
        img = Image(img)
    segmentation_mask_necessary = mask_cells or window_size is None
    if segmentation_mask is None and segmentation_mask_necessary:
        raise ValueError("Segmentation mask is necessary when masking cells or window size is not given.")
    if window_size is None:
        logging.info("Finding window size")
        window_size = find_bbox_size(segmentation_mask)
        logging.info(f"Window size automatically set to {window_size}")
        window_size = (window_size, window_size)
    destination = pathlib.Path(destination)
    if channels is None:
        channels = np.arange(img.n_channels)
    else:
        if len(channels) != len(set(channels)):
            vals, counts = np.unique(channels, return_counts=True)
            warnings.warn(DuplicateChannelWarning(duplicate_channels=vals[counts > 1]))
        channels = np.array(channels)
    if np.min(channels) < 0 or np.max(channels) >= img.n_channels:
        raise ValueError(f"Channel indices must be between 0 and {img.n_channels - 1}.")
    array_shape = (len(channels), roi_data.shape[0], window_size[0], window_size[1])
    if cells_per_chunk is None:
        array_chunks = find_chunk_size(
            array_shape, np.dtype(img.dtype).itemsize, target_bytes=target_chunk_size
        )
    else:
        array_chunks = tuple(
            int(x) for x in (len(channels), cells_per_chunk, window_size[0], window_size[1])
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
            shape=(roi_data.shape[0], window_size[0], window_size[1]),
            dtype=np.bool_,
            compressor=Blosc(cname="zstd", clevel=2, shuffle=Blosc.SHUFFLE),
            chunks=(array_chunks[1], array_chunks[2], array_chunks[3]),
        )
        mask_thumbnails_temp = np.empty(
            (roi_data.shape[0], window_size[0], window_size[1]),
            dtype=np.bool_,
        )
        cut_cells(
            segmentation_mask,
            roi_data,
            mask_thumbnails_temp,
            create_mask_thumbnails=True,
        )
        mask_thumbnails[...] = mask_thumbnails_temp[...]
        # If writing to zip files was requested zipping up the directory now
        if use_zip:
            logging.debug(f"Zipping up mask to {destination_mask}")
            zip_dir(
                mask_store.path,
                destination_mask,
            )
    n_cells = array_shape[1]
    # Only load required channels
    if img.n_channels == 1:
        var = img.base_series.shape[:]
    else:
        var = img.base_series.shape[1:]
    img_shape = np.array((len(channels),) + var, dtype=np.int64)
    n_bytes_img = int(img.dtype.itemsize * np.prod(img_shape))
    n_bytes_mask = (
        int(
            mask_thumbnails.dtype.itemsize
            * np.prod(np.array(mask_thumbnails.shape, dtype=np.int64))
        )
        if mask_thumbnails is not None
        else 0
    )
    n_bytes_total = n_bytes_img + n_bytes_mask
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=processes
    ) as executor, SharedMemoryManager() as smm:
        if n_bytes_total < cache_size:
            logging.info(
                f"Image size ({round(n_bytes_total / 1024**2)} MB) is smaller "
                f"than cache size ({round(cache_size / 1024**2)} MB). "
                "Loading entire image into memory."
            )
            raw_sm = smm.SharedMemory(size=n_bytes_img)
            img_array = np.ndarray(img_shape, dtype=img.dtype, buffer=raw_sm.buf)

            if img.n_channels == 1:
                img_array[...] = img.zarr_no_cache.oindex[:, :]
            else:
                img_array[...] = img.zarr_no_cache.oindex[channels, :, :]

            if mask_thumbnails is not None:
                raw_sm_mask = smm.SharedMemory(size=n_bytes_mask)
                mask_thumbnails_array = np.ndarray(
                    mask_thumbnails.shape,
                    dtype=mask_thumbnails.dtype,
                    buffer=raw_sm_mask.buf,
                )
                mask_thumbnails_array[...] = mask_thumbnails[...]
            futures = {
                executor.submit(
                    cut_cell_range_shared_mem,
                    SharedNumpyArraySpec(raw_sm.name, img_array.shape, img_array.dtype),
                    cell_data=roi_data,
                    cell_range=cell_range,
                    window_size=window_size,
                    cut_array=file,
                    mask_thumbnails_spec=SharedNumpyArraySpec(
                        raw_sm_mask.name, mask_thumbnails.shape, mask_thumbnails.dtype
                    )
                    if mask_thumbnails is not None
                    else None,
                ): cell_range
                for cell_range in pairwise(range_all(0, n_cells, array_chunks[1]))
            }
        else:
            logging.warn(
                f"Image size ({round(n_bytes_total / 1024**2)} MB) is larger than "
                f"cache size ({round(cache_size / 1024**2)} MB). "
                "This results in many reads from disk and will be slow. Consider "
                "increasing the cache size."
            )
            if processes > 1:
                logging.warn("Processing in parallel may be slow with small caches. Consider using -p 1.")
            futures = {
                executor.submit(
                    cut_cell_range,
                    img.path,
                    cell_data=roi_data,
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
                logging.info(f"Tiles {cell_range[0]}-{cell_range[1]} done")
            except Exception as ex:
                logging.error(
                    f"Error processing tiles {cell_range[0]}-{cell_range[1]}: {ex}"
                )
                raise ex
    if use_zip:
        logging.info("Zipping up thumbnails")
        logging.debug(f"Zipping up to {destination}")
        zip_dir(store.path, destination)
