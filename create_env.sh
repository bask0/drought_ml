
#!/bin/bash

echo ">>> Installing conda packages"


# if it breaks it is tensorboard
conda create --yes --name dml python=3.10 cdo nco tmux tensorboard jupyterlab cartopy matplotlib gxx_linux-64==11.1.0

eval "$(conda shell.bash hook)"
conda activate dml

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install pytorch-lightning
# if it breaks it is tensorflow
pip install seaborn zarr netcdf4 xarray dask scikit-gstat test-tube flake8 tensorflow captum scikit-learn prettytable 'jsonargparse[signatures]'

pip install -e .

echo ">>> Done!"
