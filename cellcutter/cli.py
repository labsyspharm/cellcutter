import argparse
import logging
import os

import numpy as np
import pandas as pd

from . import cut as cut_mod


def cut():
    parser = argparse.ArgumentParser(
        description="""Cut out thumbnail images of all cells.

        Thumbnails will be stored as Zarr array (https://zarr.readthedocs.io/en/stable/index.html)
        with dimensions [#channels, #cells, window_size, window_size].

        The chunking shape greatly influences performance
        https://zarr.readthedocs.io/en/stable/tutorial.html#chunk-optimizations.
        """,
    )
    parser.add_argument(
        "IMAGE",
        help="Path to image in TIFF format, potentially with multiple channels. "
        "Thumbnails will be created from each channel.",
    )
    parser.add_argument(
        "SEGMENTATION_MASK",
        help="Path to segmentation mask image in TIFF format. "
        "Used to automatically chose window size and find cell outlines.",
    )
    parser.add_argument(
        "CELL_DATA",
        help="Path to CSV file with a row for each cell. Must contain columns CellID "
        "(must correspond to the cell IDs in the segmentation mask), Y_centroid, and X_centroid "
        "(the coordinates of cell centroids).",
    )
    parser.add_argument(
        "DESTINATION",
        help="Path to a new directory where cell thumbnails will be stored in Zarr format.",
    )
    parser.add_argument(
        "-p", default=1, type=int, help="Number of processes run in parallel.",
    )
    parser.add_argument(
        "-z",
        default=False,
        action="store_true",
        help="Store thumbnails in a single zip file instead of a directory.",
    )
    parser.add_argument(
        "--window-size",
        default=None,
        type=int,
        help="Size of the cell thumbnail in pixels. Defaults to size of largest cell.",
    )
    parser.add_argument(
        "--mask-cells",
        default=False,
        action="store_true",
        help="Fill every pixel not occupied by the target cell with zeros.",
    )
    chunk_size_group = parser.add_mutually_exclusive_group()
    chunk_size_group.add_argument(
        "--chunk-size",
        default=32,
        type=int,
        help="Desired uncompressed chunk size in MB. "
        "(See https://zarr.readthedocs.io/en/stable/tutorial.html#chunk-optimizations) "
        "Since the other chunk dimensions are fixed as [#channels, #cells, window_size, window_size], "
        "this argument determines the number of cells per chunk. (Default: 32 MB)",
    )
    chunk_size_group.add_argument(
        "--cells-per-chunk",
        default=None,
        type=int,
        help="Desired number of cells stored per Zarr array chunk. By default this is "
        "determined automatically using the chunk size parameter. Setting this option "
        "overrides the chunk size parameter.",
    )
    parser.add_argument(
        "--cache-size",
        default=10 * 1024,
        type=int,
        help="Cache size for reading image tiles in MB. For best performance the cache "
        "size should be larger than the size of the image. "
        "(Default: 10240 MB = 10 GB)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        default=None,
        help="Indices of channels (1-based) to include in the output e.g., --channels 1 3 5. "
        "Default is to include all channels. This option must be *after* all positional arguments.",
    )
    args = parser.parse_intermixed_args()
    logging.basicConfig(
        format="%(processName)s %(asctime)s %(levelname)s: %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    logging.info(args)
    img = cut_mod.Image(args.IMAGE)
    segmentation_mask_img = cut_mod.Image(args.SEGMENTATION_MASK)
    logging.info("Loading cell data")
    cell_data_df = pd.read_csv(
        args.CELL_DATA, usecols=["CellID", "X_centroid", "Y_centroid"]
    )
    logging.info(f"Found {len(cell_data_df)} cells")
    channels = None
    if args.channels:
        channels = np.unique(np.array(args.channels)) - 1
        if channels[0] < 0 or channels[-1] >= img.n_channels:
            raise ValueError(f"Channel indices must be between 1 and {img.n_channels}.")
    cut_mod.process_all_channels(
        img,
        segmentation_mask_img,
        cell_data_df,
        args.DESTINATION,
        window_size=args.window_size,
        mask_cells=args.mask_cells,
        processes=args.p,
        target_chunk_size=args.chunk_size * 1024 * 1024,
        cells_per_chunk=args.cells_per_chunk,
        channels=channels,
        use_zip=args.z,
        cache_size=args.cache_size * 1024 * 1024,
    )
    logging.info("Done")
