[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Create cell thumbnails

Module for creating thumbnail images of all cells given a image and segmentation mask.

## Installation

```
pip install git+https://github.com/labsyspharm/cellcutter.git#egg=cellcutter
```

## Usage

```
cut_cells [OPTIONS] IMAGE SEGMENTATION_MASK CELL_DATA DESTINATION

  Cut out thumbnail images of all cells.

  IMAGE - Path to image in TIFF format, potentially with multiple channels.
  Thumbnails will be created from each channel.

  SEGMENTATION_MASK - Path to segmentation mask image in TIFF format. Used to
  automatically chose window size and find cell outlines.

  CELL_DATA - Path to CSV file with a row for each cell. Must contain columns
  CellID (must correspond to the cell IDs in the segmentation mask),
  Y_centroid, and X_centroid.

  DESTINATION - Path to a new directory where cell thumbnails will be stored
  in Zarr format (https://zarr.readthedocs.io/en/stable/index.html).

  The output is a Zarr array with the dimensions [#channels, #cells,
  window_size, window_size]

Options:
  --window-size INTEGER           Size of the cell thumbnail in pixels.
                                  Defaults to size of largest cell.
  --mask-cells / --dont-mask-cells
                                  Fill every pixel not occupied by the target
                                  cell with zeros.
  --help                          Show this message and exit.
```

## Example

```
cut_cells exemplar-001/registration/exemplar-001.ome.tif \
  exemplar-001/segmentation/unmicst-exemplar-001/cellMask.tif \
  exemplar-001/quantification/unmicst-exemplar-001_cellMask.csv \
  cellMaskThumbnails.zarr
```

## Funding

This work is supported by the following:

* NCI grants U54-CA22508U2C-CA233262 and U2C-CA233280
* NIH grant 1U54CA225088: Systems Pharmacology of Therapeutic and Adverse Responses to Immune Checkpoint and Small Molecule Drugs
