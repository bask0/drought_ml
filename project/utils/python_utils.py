
import logging 
from typing import Any


class LoggerAction(object):
    def __init__(self, level: int = logging.DEBUG, logger_name: str = 'pytorch_lightning'):
        self.level = level
        self.logger_name = logger_name

    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        if self.level >= logger.level:
            return logger
        else:
            None

    def __exit__(self, type, value, traceback):
        pass


class generic:
    pass


class Sampler(object):
    def __init__(self, type: type, *choices: Any):
        self.type = type
        self.choices = list(choices)

    def __call__(self, rs: np.random.RandomState) -> Any:
        if self.type is generic:
            idx = rs.choice(len('verbose:'))
            return self.choices[idx]
        else:
            return self.type(rs.choice(self.choices))

    def __repr__(self) -> str:
        s = f'Sampler(type={self.type.__name__}, choices={self.choices})'
        return s


class SearchSpace:
    def __init__(self, search_space: dict, seed: Optional[int] = None, max_iter: int = -1):
        self.search_space = search_space
        self.seed = seed
        self.rs = np.random.RandomState(seed=seed)
        self.max_iter = np.inf if max_iter == -1 else max_iter
        self.current_iter = 0

    @staticmethod
    def _new_sampler(type: type, *choices: Any) -> Sampler:
        if not isinstance(choices, (list, tuple)):
            raise TypeError(
                f'`choices` must be a list or a tuple, is {type(choices).__name__}.'
            )

        return Sampler(type, *choices)

    @classmethod
    def sample_int(cls, *choices: int) -> Sampler:
        return cls._new_sampler(int, *choices)

    @classmethod
    def sample_float(cls, *choices: float) -> Sampler:
        return cls._new_sampler(float, *choices)

    @classmethod
    def sample_str(cls, *choices: str) -> Sampler:
        return cls._new_sampler(str, *choices)

    @classmethod
    def sample_bool(cls, *choices: bool) -> Sampler:
        return cls._new_sampler(bool, *choices)

    @classmethod
    def sample_generic(cls, *choices: generic) -> Sampler:
        return cls._new_sampler(generic, *choices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_iter >= self.max_iter:
            raise StopIteration
        else:
            self.current_iter += 1
            if self.search_space is not None:
                return self._sample_from_search_space(self.search_space)
            else:
                return None

    def _sample_from_search_space(self, el: dict) -> Any:
        hps = {}
        if isinstance(el, dict):
            for k, v in el.items():
                if isinstance(v, dict):
                    hps[k] = self._sample_from_search_space(v)
                elif isinstance(v, Sampler):
                    hps[k] = v(self.rs)
                else:
                    raise TypeError(
                        f'`search_space` contains elements that are not dicts or Samplers, but `{type(v).__name__}`.'
                    )
        else:
            raise TypeError(
                f'`el` must be a dictionary, is `{type(el).__name__}`.'
            )

        return hps

    def __repr__(self):
        s = 'SearchSpace'
        s += f'\n  - seed={self.seed}'
        s += f'\n  - max_iter={self.max_iter}'
        s += '\n  - search_space:'
        if self.search_space is not None:
            for k, v in self.search_space.items():
                s += f'\n    - {k}: {v}'
        else:
            s += f'\n    - empty'
        return s
