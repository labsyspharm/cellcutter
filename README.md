[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Create cell thumbnails

Module for creating thumbnail images of all cells given a image and segmentation mask.

## Installation

```
pip install git+https://github.com/labsyspharm/cellcutter.git#egg=cellcutter
```

## Usage

```
cut_cells [-h] [-p P] [--window-size WINDOW_SIZE]
                 [--mask-cells | --dont-mask-cells] [--chunk-size CHUNK_SIZE]
                 IMAGE SEGMENTATION_MASK CELL_DATA DESTINATION
                 [--channels [CHANNELS ...]]

Cut out thumbnail images of all cells. Thumbnails will be stored as Zarr array
(https://zarr.readthedocs.io/en/stable/index.html) with dimensions [#channels,
#cells, window_size, window_size].

positional arguments:
  IMAGE                 Path to image in TIFF format, potentially with
                        multiple channels. Thumbnails will be created from
                        each channel.
  SEGMENTATION_MASK     Path to segmentation mask image in TIFF format. Used
                        to automatically chose window size and find cell
                        outlines.
  CELL_DATA             Path to CSV file with a row for each cell. Must
                        contain columns CellID (must correspond to the cell
                        IDs in the segmentation mask), Y_centroid, and
                        X_centroid.
  DESTINATION           Path to a new directory where cell thumbnails will be
                        stored in Zarr format.

optional arguments:
  -h, --help            show this help message and exit
  -p P                  Number of processes run in parallel.
  -z                    Store thumbnails in a single zip file instead of a
                        directory. (default: False)
  --window-size WINDOW_SIZE
                        Size of the cell thumbnail in pixels. Defaults to size
                        of largest cell.
  --mask-cells, --dont-mask-cells
                        Fill every pixel not occupied by the target cell with
                        zeros. (Default: --mask-cells) (default: True)
  --chunk-size CHUNK_SIZE
                        Desired uncompressed chunk size in MB.See https://zarr
                        .readthedocs.io/en/stable/tutorial.html#chunk-
                        optimizations. (Default: 32)
  --channels [CHANNELS ...]
                        Indices of channels (1-based) to include in the output
                        e.g., --channels 1 3 5. Default is to include all
                        channels. This option has to *after* the positional
                        arguments.
```

## Example

Example data are available for [download at mcmicro.org](https://mcmicro.org/datasets.html).

```
cut_cells exemplar-001/registration/exemplar-001.ome.tif \
  exemplar-001/segmentation/unmicst-exemplar-001/cellMask.tif \
  exemplar-001/quantification/unmicst-exemplar-001_cellMask.csv \
  cellMaskThumbnails.zarr
```

## Reading the zarr array output

```python
import zarr
from matplotlib import pyplot as plt
```

```python
x = zarr.open("cellMaskThumbnails.zarr", mode = "r")
```

    <zarr.core.Array (12, 9522, 46, 46) uint16 read-only>

```python
plt.imshow(x[0, 0, ...])
```

![png](docs/assets/example_thumbnail.png)

```python
plt.figure(figsize=(10, 10))
for i in range(64):
    ax = plt.subplot(8, 8, i + 1)
    ax.axis("off")
    ax.imshow(x[0, i, ...])
plt.tight_layout()
```

![png](docs/assets/example_thumbnails.png)

## Funding

This work is supported by the following:

* NCI grants U54-CA22508U2C-CA233262 and U2C-CA233280
* NIH grant 1U54CA225088: Systems Pharmacology of Therapeutic and Adverse Responses to Immune Checkpoint and Small Molecule Drugs
