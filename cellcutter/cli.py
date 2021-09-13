import argparse
import logging

import numpy as np
import pandas as pd

from . import cut as cut_mod


class BooleanOptionalAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
        prefix="--dont-",
    ):
        self.prefix = prefix
        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith("--"):
                option_string = prefix + option_string[2:]
                _option_strings.append(option_string)
        if help is not None and default is not None:
            help += f" (default: {default})"
        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith(self.prefix))

    def format_usage(self):
        return " | ".join(self.option_strings)


def cut():
    parser = argparse.ArgumentParser(
        description="""Cut out thumbnail images of all cells.

        Thumbnails will be stored as Zarr array (https://zarr.readthedocs.io/en/stable/index.html)
        with dimensions [#channels, #cells, window_size, window_size].
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
        "(must correspond to the cell IDs in the segmentation mask), Y_centroid, and X_centroid.",
    )
    parser.add_argument(
        "DESTINATION",
        help="Path to a new directory where cell thumbnails will be stored in Zarr format.",
    )
    parser.add_argument(
        "-p", default=1, type=int, help="Number of processes run in parallel.",
    )
    parser.add_argument(
        "--window-size",
        default=None,
        type=int,
        help="Size of the cell thumbnail in pixels. Defaults to size of largest cell.",
    )
    parser.add_argument(
        "--mask-cells",
        default=True,
        action=BooleanOptionalAction,
        help="Fill every pixel not occupied by the target cell with zeros. (Default: --mask-cells)",
    )
    parser.add_argument(
        "--chunk-size",
        default=32,
        type=int,
        help="Desired uncompressed chunk size in MB."
        "See https://zarr.readthedocs.io/en/stable/tutorial.html#chunk-optimizations. (Default: 32)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        default=None,
        help="Indices of channels (1-based) to include in the output e.g., --channels 1 3 5. "
        "Default is to include all channels. This option has to *after* the positional arguments.",
    )
    args = parser.parse_intermixed_args()
    logging.basicConfig(
        format="%(processName)s %(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
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
        channels=channels,
    )
    logging.info("Done")
