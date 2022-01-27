import logging
import concurrent.futures
import itertools
import pathlib
import zipfile
from typing import Optional, Union, Iterable

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


def zip_dir(dir: Union[pathlib.Path, str], zip_file: Union[pathlib.Path, str]) -> None:
    "Zip the contents of the given directory into a ZIP file."
    dir = pathlib.Path(dir)
    # Don't compress the ZIP file. Zarr arrays are already compressed.
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in dir.rglob("*"):
            zf.write(f, f.relative_to(dir))


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
    zarr_channel_index: Optional[int] = None,
) -> None:
    "Cut cells from a given image, in chunks aligned with the output array, and write them into the given Zarr or Numpy array."
    cell_chunk_size = (
        cell_stack.chunks[1] if len(cell_stack.chunks) == 4 else cell_stack.chunks[0]
    )
    n_cells = len(cell_data)
    window_size_h = window_size // 2
    img = np.pad(img, ((window_size_h, window_size_h), (window_size_h, window_size_h)))
    # Iterate over slices of the data equal to the chunk size in the cell dimension
    for s, e in pairwise(
        itertools.chain(range(0, n_cells, cell_chunk_size), [n_cells])
    ):
        if s % 100000 <= cell_chunk_size:
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
        if zarr_channel_index is not None:
            cell_stack[zarr_channel_index, s:e, ...] = cell_stack_c
        else:
            cell_stack[s:e, ...] = cell_stack_c


def cut_cells_mp(
    image: str,
    cell_data: pd.DataFrame,
    channel_index: int,
    window_size: int,
    cut_array: zarr.Array,
    mask_thumbnails: Optional[zarr.Array] = None,
    zarr_channel_index: Optional[int] = None,
) -> None:
    "Load single channel from the given TIFF file and cut out cells."
    logging.info(f"Loading channel {channel_index}")
    img = Image(image).get_channel(channel_index)
    if zarr_channel_index is None:
        zarr_channel_index = channel_index
    cut_cells_chunked(
        img,
        cell_data,
        window_size,
        cut_array,
        dtype=img.dtype,
        mask_thumbnails=mask_thumbnails,
        zarr_channel_index=zarr_channel_index,
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
    destination: Union[pathlib.Path, str],
    window_size: Optional[int],
    mask_cells: bool = True,
    processes: int = 1,
    target_chunk_size: int = 32 * 1024 * 1024,
    channels: Optional[Iterable[int]] = None,
    use_zip: bool = False,
) -> None:
    "Given an image, segmentation mask, and cell positions, cut out cells and write a stack of cell thumbnails in Zarr format."
    destination = pathlib.Path(destination)
    if use_zip:
        destination = destination.with_suffix(".zip")
    logging.info("Loading segmentation mask")
    segmentation_mask_img = segmentation_mask.get_channel(0)
    # Check if all cell IDs present in the CSV file are also represented in the segmentation mask
    logging.info("Check consistency of cell IDs")
    cell_ids_in_segmentation_mask = np.unique(segmentation_mask_img)
    n_not_in_segmentation_mask = set(cell_data["CellID"]) - set(
        cell_ids_in_segmentation_mask
    )
    if len(n_not_in_segmentation_mask) > 0:
        raise ValueError(
            f"{len(n_not_in_segmentation_mask)} cell IDs in the CELL_DATA CSV file are not present in the segmentation mask."
        )
    # Remove cells from segmentation mask that are not present in the CSV
    segmentation_mask_img[~np.isin(segmentation_mask_img, cell_data["CellID"])] = 0
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
    array_dtype = img.get_channel(0).dtype
    array_chunks = find_chunk_size(
        array_shape, np.dtype(array_dtype).itemsize, target_size=target_chunk_size
    )
    logging.info(f"Using chunks of shape {array_chunks}")
    # If writing to a zip file, create a temporary directory to store the zarr files
    # and compress them into the zip file at the end. Solves issues with concurrent
    # access to the same zip file.
    store = zarr.DirectoryStore(str(destination)) if not use_zip else zarr.TempStore()
    logging.debug(f"Writing thumbnails to {store.path}")
    # Chosing low compression level for speed. Size difference is negligible.
    file = zarr.create(
        store=store,
        overwrite=True,
        shape=array_shape,
        dtype=array_dtype,
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
        cut_cells_chunked(
            segmentation_mask_img,
            cell_data,
            window_size,
            mask_thumbnails,
            dtype=np.bool_,
            create_mask_thumbnails=True,
        )
        # If writing to zip files was requested zipping up the directory now
        if use_zip:
            logging.debug(f"Zipping up mask to {destination_mask}")
            zip_dir(
                mask_store.path, destination_mask,
            )
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        futures = {
            executor.submit(
                cut_cells_mp,
                img.path,
                cell_data,
                c,
                window_size,
                file,
                mask_thumbnails=mask_thumbnails,
                zarr_channel_index=i,
            ): c
            for i, c in enumerate(channels)
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                future.result()
                logging.info(f"Channel {i} done")
            except Exception as ex:
                logging.error(f"Error processing channel {i}: {ex}")
    if use_zip:
        logging.info("Zipping up thumbnails")
        logging.debug(f"Zipping up to {destination}")
        zip_dir(store.path, destination)
