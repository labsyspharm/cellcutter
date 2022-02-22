import logging
import concurrent.futures
import itertools
import pathlib
from tkinter import W
import zipfile
from typing import Optional, Union, Iterable, Tuple

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

def padded_subset(img: Union[zarr.Array, np.ndarray], x: int, y: int, window_size: Tuple[int, int]) -> np.ndarray:
    "Return a padded subset of the image with the given window size"
    wh = img.shape[-2:]
    s = (
        (x - window_size[0] // 2, x + window_size[0] // 2),
        (y - window_size[1] // 2, y + window_size[1] // 2),
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
    out = np.zeros(
        img.shape[:-2] + (window_size[0], window_size[1]), dtype=img.dtype
    )
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
        self.base_series.pages[0].dtype


def cut_cells(
    img: Union[zarr.Array, np.ndarray],
    cell_data: pd.DataFrame,
    window_size: int,
    cell_stack: Union[zarr.Array, np.ndarray],
    mask_thumbnails: Optional[Union[zarr.Array, np.ndarray]] = None,
    create_mask_thumbnails: bool = False,
) -> None:
    "Cut cells from a given image and write them into the given Zarr or Numpy array."
    for i, c in enumerate(cell_data.itertuples()):
        centroids = np.array([c.Y_centroid, c.X_centroid]).astype(int)
        thumbnail = padded_subset(
            img, centroids[0], centroids[1], window_size=(window_size, window_size)
        )
        if create_mask_thumbnails:
            thumbnail = thumbnail == c.CellID
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
) -> None:
    "Load single channel from the given TIFF file and cut out cells."
    image = Image(image, cache_size=cache_size)
    cell_data_subset = cell_data.iloc[cell_range[0] : cell_range[1]]
    cell_stack_temp = np.empty(
        (image.n_channels, cell_range[1] - cell_range[0], window_size, window_size), dtype=image.dtype
    )
    cut_cells(image.zarr, cell_data_subset, window_size, cell_stack_temp, mask_thumbnails)
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
    array_chunks = find_chunk_size(
        array_shape, np.dtype(img.dtype).itemsize, target_bytes=target_chunk_size
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
        cut_cells(
            segmentation_mask_img,
            cell_data,
            window_size,
            mask_thumbnails,
            create_mask_thumbnails=True,
        )
        # If writing to zip files was requested zipping up the directory now
        if use_zip:
            logging.debug(f"Zipping up mask to {destination_mask}")
            zip_dir(
                mask_store.path, destination_mask,
            )
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        n_cells = array_shape[1]
        futures = {
            executor.submit(
                cut_cell_range,
                img.path,
                cell_data=cell_data,
                cell_range=cell_range,
                window_size=window_size,
                cut_array=file,
                mask_thumbnails=mask_thumbnails,
                cache_size=cache_size,
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
