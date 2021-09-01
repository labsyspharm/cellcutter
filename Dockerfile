FROM continuumio/miniconda3
RUN conda install click scikit-image numpy pandas tifffile zarr numcodecs
COPY . /app/
RUN python -m pip install /app/
ENTRYPOINT ["cut_cells"]
