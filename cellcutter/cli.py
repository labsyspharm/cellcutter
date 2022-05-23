import argparse
import logging
import os
import shutil
import warnings

import numpy as np
import pandas as pd

from . import cut as cut_mod


def formatwarning_duplicate_channel(*args, **kwargs):
    if isinstance(args[0], cut_mod.DuplicateChannelWarning):
        args[0].duplicate_channels += 1
    return str(args[0])


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
        "(the coordinates of cell centroids). "
        "Only cells represented in the given CSV file will be used, even if "
        "additional cells are present in the segmentation mask. Cells are written to "
        "the Zarr array in the same order as they appear in the CSV file.",
    )
    parser.add_argument(
        "DESTINATION", metavar="DESTINATION.zarr",
        help="Path to a new directory where cell thumbnails will be stored in Zarr format. "
        "Must end in '.zarr' and must not already exist. These restrictions can be lifted by "
        "using the '-f/--force' flag.",
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
        "-f", "--force",
        default=False,
        action="store_true",
        help="Overwrite existing destination directory and don't enforce '.zarr' extension.",
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
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        default=None,
        help="Indices of channels (1-based) to include in the output e.g., --channels 1 3 5. "
        "Channels are included in the file in the given order. If not specified, by default all channels are included. "
        "This option must be *after* all positional arguments.",
    )
    parser.add_argument(
        "--cache-size",
        default=10 * 1024,
        type=int,
        help="Cache size for reading image tiles in MB. For best performance the cache "
        "size should be larger than the size of the image. "
        "(Default: 10240 MB = 10 GB)",
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

    args = parser.parse_intermixed_args()
    logging.basicConfig(
        format="%(processName)s %(asctime)s %(levelname)s: %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    logging.captureWarnings(True)
    logging.info(args)
    if not args.DESTINATION.endswith(".zarr") and not args.force:
        parser.error(
            "Destination must end in '.zarr' unless '-f/--force' is used."
        )
    if os.path.exists(args.DESTINATION):
        if not args.force:
            parser.error(
                "Destination directory already exists. Use '-f/--force' to overwrite."
            )
        else:
            # Preemptively remove destination directory or file
            if os.path.isfile(args.DESTINATION):
                os.remove(args.DESTINATION)
            else:
                shutil.rmtree(args.DESTINATION)
    img = cut_mod.Image(args.IMAGE)
    segmentation_mask_img = cut_mod.Image(args.SEGMENTATION_MASK)
    logging.info("Loading cell data")
    cell_data_df = pd.read_csv(
        args.CELL_DATA, usecols=["CellID", "X_centroid", "Y_centroid"]
    )
    logging.info(f"Found {len(cell_data_df)} cells")
    channels = None
    if args.channels:
        # CLI uses 1-based indices, but we use 0-based indices internally
        channels = np.array(args.channels) - 1
        if np.min(channels) < 0 or np.max(channels) >= img.n_channels:
            parser.error(f"Channel indices must be between 1 and {img.n_channels}.")
    with warnings.catch_warnings():
        # Make sure warning about duplicate channels are printed using 1-based indices
        warnings.formatwarning = formatwarning_duplicate_channel
        cut_mod.process_image(
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
