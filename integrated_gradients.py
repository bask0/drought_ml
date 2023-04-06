
from argparse import ArgumentParser
import xarray as xr
import zarr
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import torch
from torch import Tensor
import yaml
from yaml import SafeLoader
from captum.attr import IntegratedGradients
import tqdm
import importlib.util
import os
import sys
from time import sleep
from typing import Callable
from numpy.typing import ArrayLike
import logging

from project.dataset import GeoDataQueue, QueueFiller
from project.utils.types import BatchPattern

torch.backends.cudnn.enabled = False


logger = logging.getLogger(__name__)


class IGWriter(object):
    def __init__(self, zarr_path: str, dataloader: torch.utils.data.DataLoader):
        logger.debug('Initializing IGWriter.')
        self.zarr_path = zarr_path

        self.dataloader = dataloader
        self.mask = self.dataloader.dataset.mask.load()

        self.chunks = {}
        self.chunk_bounds_lat = QueueFiller.coords2bounds(dataloader.dataset.data.mask, dim='lat', chunk_size=20)
        self.chunk_bounds_lon = QueueFiller.coords2bounds(dataloader.dataset.data.mask, dim='lon', chunk_size=20)

        self.zarr_file = None

    def write_attr(self, attr_hourly: ArrayLike, attr_static: ArrayLike, batch: BatchPattern):

        if self.zarr_file is None:
            self.zarr_file = xr.open_zarr(self.zarr_path)

        chunk_ids = batch.coords.chunk.cpu().numpy()
        latitudes = batch.coords.lat.cpu().numpy()
        longitudes = batch.coords.lon.cpu().numpy()
        u = ' | '.join([f'chunk_id {u:d}: n={n:d}' for u, n in zip(*np.unique(chunk_ids, return_counts=True))])
        logger.debug(f'Adding attributions: {u}.')

        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id not in self.chunks:
                logger.debug(
                    f'Adding new chunk with ID {chunk_id}'
                )
                chunk_lat, chunk_lon = self.dataloader.dataset.chunk_coords[chunk_id]

                lat_bounds = slice(*self.chunk_bounds_lat[chunk_lat])
                lon_bounds = slice(*self.chunk_bounds_lon[chunk_lon])
                num_samples = self.mask.isel(lat=lat_bounds, lon=lon_bounds).sum().compute().item()

                chunk_ds = self.zarr_file.isel(lat=lat_bounds, lon=lon_bounds).load()

                self.chunks[chunk_id] = {
                    'ds': chunk_ds,
                    'num_samples': num_samples,
                    'num_saved': 0,
                    'lat_bounds': lat_bounds,
                    'lon_bounds': lon_bounds
                }
                logger.debug(
                    f'Done adding new chunk with ID {chunk_id}. Current chunks '
                    f'loaded {list(self.chunks.keys())} (={len(self.chunks)}).'
                )

            for v, var in enumerate(self.dataloader.dataset.features_hourly):
                for i in range(attr_hourly.shape[0]):
                    lat = self.chunks[chunk_id]['ds'].lat.sel(lat=latitudes[i], method='nearest').item()
                    lon = self.chunks[chunk_id]['ds'].lon.sel(lon=longitudes[i], method='nearest').item()

                    self.chunks[chunk_id]['ds'][var].loc[
                        {
                            'lat': lat,
                            'lon': lon,
                        }
                    ] = attr_hourly[i, ..., v]

            for v, var in enumerate(self.dataloader.dataset.features_static):
                for i in range(attr_static.shape[0]):
                    lat = self.chunks[chunk_id]['ds'].lat.sel(lat=latitudes[i], method='nearest').item()
                    lon = self.chunks[chunk_id]['ds'].lon.sel(lon=longitudes[i], method='nearest').item()

                    self.chunks[chunk_id]['ds'][var].loc[
                        {
                            'lat': lat,
                            'lon': lon,
                        }
                    ] = attr_static[i, ..., v]

            self.chunks[chunk_id]['num_saved'] += 1

            if self.chunks[chunk_id]['num_saved'] == self.chunks[chunk_id]['num_samples']:
                logger.debug(f'Writing chunk {chunk_id} to file.')
                self.write_chunk(self.chunks[chunk_id])
                del self.chunks[chunk_id]
                logger.debug(f'Done writing chunk {chunk_id} to file. {len(self.chunks)} loaded currently.')

        logger.debug(f'Churrent chunks: {self.chunk_summary()}.')

    def finalize(self) -> None:

        for chunk in self.chunks.values():
            self.write_chunk(chunk)

        self.chunks = {}

    def write_chunk(self, chunk: dict) -> None:
        chunk_ds = chunk['ds'].drop_vars(['time', 'context', 'hour'])

        chunk_ds.to_zarr(self.zarr_file.encoding['source'], consolidated=False, region={
            'lat': chunk['lat_bounds'],
            'lon': chunk['lon_bounds']
        })

    def chunk_summary(self) -> str:
        s = []

        for key, val in self.chunks.items():
            s.append(f'ID {key}: {val["num_saved"]} / {val["num_samples"]}')
        return ' | '.join(s)


class Attributions(object):
    def __init__(
            self,
            model_call: Callable,
            init_batch: BatchPattern,
            geodata: GeoDataQueue,
            context_size: int,
            start_date: str,
            end_date: str,
            device: torch.device,
            ig_writer: IGWriter,
            zarr_dir: str) -> None:

        self.model_call = model_call
        self.window_start = init_batch.coords.window_start[0]
        self.window_end = init_batch.coords.window_end[0]
        self.geodata = geodata

        self._arg_check(init_batch, context_size, start_date)

        self.start_date = start_date
        self.context_start_date = pd.date_range(
            end=self.start_date,
            freq='D',
            periods=context_size)[0].strftime('%Y-%m-%d')
        self.end_date = end_date
        self.context_size = context_size
        self.data_start_date = pd.date_range(
            end=self.window_end,
            freq='D',
            periods=init_batch.f_hourly.shape[1])[0].strftime('%Y-%m-%d')

        self.ig_writer = ig_writer
        self.zarr_dir = zarr_dir
        self.ig = IntegratedGradients(self.model_call)

        if pd.Timestamp(self.context_start_date) < pd.Timestamp(self.data_start_date):
            raise ValueError(
                f'`context_size={context_size}` too large for `start_date={start_date}` and infered '
                f'` data_start_date={self.data_start_date}`.'
            )

        self.start_idx = len(pd.date_range(self.data_start_date, self.start_date, freq='D')) - 1
        self.end_idx = len(pd.date_range(self.data_start_date, self.end_date, freq='D')) - 1

        self.device = device

        self.emptyXR()
        self.zarr_ds = xr.open_zarr(self.zarr_dir)

        self.writer_process: mp.Process | None = None

    def __len__(self) -> int:
        return self.end_idx - self.start_idx + 1

    def get_attr(self, f_hourly: Tensor, f_static: Tensor, f_hourly_bl: Tensor, f_static_bl: Tensor):
        attributions = self.ig.attribute(
            inputs=(f_hourly, f_static),
            baselines=(f_hourly_bl, f_static_bl),
            method='gausslegendre',
            return_convergence_delta=False,
            target=-1
        )

        attr = [a.detach().cpu().numpy() for a in attributions]
        attributions = None
        return attr

    def iter_batch(self, batch: BatchPattern):
        for i in range(self.start_idx, self.end_idx + 1):
            yield self.get_attr(
                batch.f_hourly[:, i - self.context_size + 1:i + 1, ...].to(self.device),
                batch.f_static.to(self.device),
                batch.f_hourly_bl[:, i - self.context_size + 1:i + 1, ...].to(self.device),
                batch.f_static_bl.to(self.device),
            )

    def writer_ready(self) -> bool:
        if self.writer_process is None:
            return True

        if self.writer_process.is_alive():
            return False
        else:
            self.writer_process.join()
            self.writer_process = None
            return True

    def get_batch_attr(self, batch: BatchPattern) -> None:
        #batch = batch2device(batch, self.device)
        attr_hourly = []
        attr_static = []

        for a in self.iter_batch(batch):
            attr_hourly.append(a[0])
            attr_static.append(a[1])

        attr_hourly = np.stack(attr_hourly, axis=2)
        attr_static = np.stack(attr_static, axis=1)

        while not self.writer_ready():
            logger.warning('Writer process busy, waiting to continue')
            sleep(20)

        self.writer_process = mp.Process(target=self.ig_writer.write_attr, args=(attr_hourly, attr_static, batch))
        self.writer_process.start()

    def attribute(self, dataloader: torch.utils.data.DataLoader):
        for batch in tqdm.tqdm(dataloader, desc='Dataloader'):
            self.get_batch_attr(batch)
        self.ig_writer.finalize()

    def _arg_check(self, batch: BatchPattern, context_size: int, start_date: str) -> None:
        for el in ['window_start', 'window_end', 'num_days']:
            batch_el = getattr(batch.coords, el)
            if not all([e == batch_el[0] for e in batch_el]):
                raise ValueError(
                    'BatchIterator expects batch with full sequences, but found different values in on of '
                    '\'batch.coords.window_start\', \'batch.coords.window_end\', \'batch.coords.num_days\'.'
                )

        if pd.Timestamp(start_date) < pd.Timestamp(self.window_start):
            raise ValueError(
                f'`start_date={start_date}` out of batch range {self.window_start, self.window_end}'
            )
        if pd.Timestamp(start_date) > pd.Timestamp(self.window_end):
            raise ValueError(
                f'`start_date={start_date}` out of batch range {self.window_start, self.window_end}'
            )

    def emptyXR(self) -> xr.Dataset:
        obs = self.geodata.ds.sel(
            time=slice(self.start_date, self.end_date))[self.geodata.features_hourly + self.geodata.features_static]

        attr = xr.full_like(obs, np.nan)
        for var in attr.data_vars:
            if 'time' in attr[var].dims:
                attr[var] = attr[var].expand_dims(context=np.arange(-self.context_size + 1, 1))
            else:
                attr[var] = attr[var].expand_dims(time=obs.time)

        lat_chunksize = obs.chunksizes['lat'][0]
        lon_chunksize = obs.chunksizes['lon'][0]
        attr = attr.chunk({
            'context': -1,
            'time': 10,
            'hour': -1,
            'lat': lat_chunksize,
            'lon': lon_chunksize
        })

        attr.to_zarr(self.zarr_dir, compute=False, mode='a')

    def finalize(self):
        self.writer_process.join()
        self.writer_process = None
        zarr.consolidate_metadata(self.zarr_dir)


class CustomManager(BaseManager):
    # nothing
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--path',
        type=str,
        help='path to experiment directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='the batch size, default is 1'
    )
    parser.add_argument(
        '--context_size',
        type=int,
        default=365 * 3,
        help='number of context days to calculate integratients for, default is 3 * 365'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='cv',
        help='run type, either `cv` (default) or `tune`'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0,
        help='the trial ID, an integer from 0 to 9, default is 0'
    )
    parser.add_argument(
        '--num_split',
        type=int,
        default=0,
        help='if > 1, the processing will be split in n chunks (use with `--split_idx`)'
    )
    parser.add_argument(
        '--split_idx',
        type=int,
        default=-1,
        help='if > 1, the processing will be split in n chunks (use )'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        help='the batch size'
    )
    parser.add_argument(
        '--logger_level',
        type=int,
        default=2,
        help='the logger level: 0=DEBUG, 1=INFO, 2:WARN, 3:ERROR, default is 2'
    )

    args = parser.parse_args()

    lvl = [
        logging.DEBUG,
        logging.INFO,
        logging.WARN,
        logging.ERROR][args.logger_level]
    logging.basicConfig(level=lvl, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    if args.type not in ['cv', 'tune']:
        raise ValueError(
            f'`type` must be one of \'cv\' or \'tune\', is {args.type}'
        )

    exp_dir = os.path.join(args.path, args.type)
    if args.type == 'cv':
        exp_config = os.path.join(exp_dir, 'best_config.yaml')
        ig_save_path = os.path.join(args.path, 'xai', 'integrated_gradients.zarr')
    else:
        exp_config = os.path.join(exp_dir, f'trial{args.trial:02d}/config.yaml')
        ig_save_path = os.path.join(exp_dir, f'trial{args.trial:02d}/ig.zarr')

    if args.trial == 0 and args.split_idx == 0:
        if os.path.exists(ig_save_path):
            raise FileExistsError(
                f'Target file \'{ig_save_path}\' already exists, remove it and run again.'
            )
    else:
        # Avoid that multiple processes create the .zarr file at the same time. Only first process
        # creates one.
        sleep(20)

    exp_model = os.path.join(exp_dir, f'trial{args.trial:02d}/checkpoints/best.ckpt')

    for p in [exp_config, exp_model]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    with open(exp_config, 'r') as f:
        config = yaml.load(f, SafeLoader)

    data_config = config['data']
    model_path = config['model']['class_path']
    model_path, model_name = model_path.rsplit('.', 1)
    model_path = model_path.replace('.', '/')
    model_path += '.py'

    spec = importlib.util.spec_from_file_location(model_name, model_path)
    model = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = model
    spec.loader.exec_module(model)
    model = getattr(model, model_name)

    data_config['batch_size'] = args.batch_size
    data_config['return_baseline'] = True
    data_config['precip_zero_baseline'] = True
    data_config['num_queue_workers'] = 1
    data_config['queue_size'] = 5
    data_config['num_split'] = args.num_split
    data_config['split_idx'] = args.split_idx
    data_config['fold_id'] = args.trial

    device = torch.device(f'cuda:{args.gpu}')

    geodata = GeoDataQueue(**data_config)
    model = model.load_from_checkpoint(
        exp_model,
        map_location=device).to(device)
    model.eval()
    dataloader = geodata.predict_dataloader()

    batch = next(iter(dataloader))

    def model_call(x, x_b):
        return model(x, x_b)[0].ano

    CustomManager.register('IGWriter', IGWriter)

    with CustomManager() as manager:
        shared_igwriter = manager.IGWriter(ig_save_path, dataloader)

        attr = Attributions(
            model_call=model_call,
            init_batch=batch,
            geodata=geodata,
            context_size=args.context_size,
            start_date='2015-01-01',
            end_date='2015-12-31',
            device=device,
            ig_writer=shared_igwriter,
            zarr_dir=ig_save_path
        )
        attr.attribute(dataloader=dataloader)
        attr.finalize()
