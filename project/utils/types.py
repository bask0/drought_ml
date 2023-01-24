
from typing import NamedTuple
from torch import Tensor


class DateLookup(NamedTuple):
    start_context_year: int
    start_window_year: int
    end_window_year: int
    start_context_date: str
    start_window_date: str
    end_window_date: str
    num_window_days : int
    time_slice: slice


class Coords(NamedTuple):
    lat: int
    lon: int
    chunk: int
    window_start: str
    window_end: str
    num_days: int


class BatchPattern(NamedTuple):
    f_hourly: Tensor
    f_static: Tensor
    t_hourly: Tensor
    t_daily: Tensor
    coords: Coords


class ReturnPattern(NamedTuple):
    mean_hat: Tensor
    var_hat: Tensor
    coords: Coords


class QueueException(NamedTuple):
    exception: Exception
    traceback: str
