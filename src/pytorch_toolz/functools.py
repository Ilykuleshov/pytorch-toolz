from typing import TypeVar, Tuple, Generic, Callable
from torch.nn import Module, Sequential
from torch import Tensor
from functools import reduce


class Parallel(Sequential):
    """A parallel container simmilar to toolz.juxt, opposite of Map.
    Applies contained layers to input, returns tuple of results. Usage same as nn.Sequential
    """
    def forward(self, input):
        return tuple(module(input) for module in self)


class _Functool(Module):
    def __init__(self, function: Callable) -> None:
        super().__init__()
        self.function = function


class Reduce(_Functool):
    """A module, simmilar to functools.reduce. Applies a two-argument function to given inputs.
    Example usage, simplified residual block:
    ```
    res_block = nn.Sequential(
        Parallel(nn.Conv2d(...), nn.Identity()),
        Reduce(torch.add)
    )
    ```
    """
    def forward(self, input: Tuple):        
        return reduce(self.function, input)


class Map(_Functool):
    """Map given function over input tuple, return resulting tuple.
    """
    def forward(self, input: Tuple):
        return tuple(map(self.function, input))


class Filter(_Functool):
    """Filter input tuple using given predicate function, return resulting tuple.
    """
    def forward(self, input: Tuple):
        return tuple(filter(self.function, input))
