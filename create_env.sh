#!/bin/bash

# conda create --name dml python=3.10 pytorch torchvision torchaudio cudatoolkit=11.3 netcdf4 xarray pandas zarr rasterio matplotlib cartopy cdo nco texlive-core dask jupyterlab ipywidgets numpy tqdm pytorch-lightning ffmpeg imageio scikit-learn optuna flake8 parallel tmux pangeo-forge-recipes python-graphviz -c pytorch conda-forge

echo ">>> Installing conda packages"

conda create --yes --name dml python=3.10 netcdf4 xarray pandas zarr rasterio matplotlib cartopy cdo nco texlive-core dask jupyterlab ipywidgets numpy tqdm pytorch-lightning ffmpeg imageio scikit-learn wandb flake8 parallel tmux python-graphviz -c pytorch -c conda-forge

echo ">>> Installing pip packages"

eval "$(conda shell.bash hook)"
conda activate dml

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

conda develop .

echo ">>> Done!"
