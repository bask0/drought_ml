
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
    dayofyear: list[int]


class VarStackPattern(NamedTuple):
    ts: Tensor | None
    msc: Tensor | None
    ano: Tensor | None
    ts_var: Tensor | None
    msc_var: Tensor | None
    ano_var: Tensor | None


class BatchPattern(NamedTuple):
    f_hourly: Tensor
    f_static: Tensor
    t_hourly_ts: Tensor
    t_hourly_msc: Tensor
    t_hourly_ano: Tensor
    t_daily_ts: Tensor
    t_daily_msc: Tensor
    t_daily_ano: Tensor
    coords: Coords


class ReturnPattern(NamedTuple):
    daily_ts: Tensor
    daily_ts_var: Tensor
    daily_msc: Tensor
    daily_msc_var: Tensor
    daily_ano: Tensor
    daily_ano_var: Tensor
    hourly_ts: Tensor
    hourly_ts_var: Tensor
    hourly_msc: Tensor
    hourly_msc_var: Tensor
    hourly_ano: Tensor
    hourly_ano_var: Tensor
    coords: Coords


class QueueException(NamedTuple):
    exception: Exception
    traceback: str
