from typing import TypeVar, Tuple, Generic
from typing import Callable
from torch.nn import Module, Sequential
from torch import Tensor
from functools import reduce


class Parallel(Sequential):
    """A parallel container simmilar to toolz.juxt, opposite of Map.
    Applies contained layers to input, returns tuple of results. Usage same as nn.Sequential
    """
    def forward(self, input):
        return tuple(module(input) for module in self)


class Reduce(Module):
    """A model, simmilar to functools.reduce. Applies a two-argument function to given inputs.
    Example usage, simplified residual block:
    ```
    res_block = nn.Sequential(
        Parallel(nn.Conv2d(...), nn.Identity()),
        Reduce(torch.add)
    )
    ```
    """
    def __init__(self, function: Callable[[Tensor, Tensor], Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, input: Tuple[Tensor, ...]) -> Tensor:
        if not isinstance(input, tuple):
            raise RuntimeError('Non-tuple input to Reduce')
        
        return reduce(self.function, input)


class Map(Module):
    """Map given function over input tuple, return resulting tuple.
    """
    def __init__(self, function: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, input: Tuple[Tensor, ...]) -> Tuple[Tensor]:
        if not isinstance(input, tuple):
            raise RuntimeError('Non-tuple input to Map')
        
        return tuple(map(self.function, input))


class Filter(Module):
    """Filter input tuple using given predicate function, return resulting tuple.
    """
    def __init__(self, function: Callable[[Tensor], bool]) -> None:
        super().__init__()
        self.function = function

    def forward(self, input: Tuple[Tensor, ...]):
        if not isinstance(input, tuple):
            raise RuntimeError('Non-tuple input to Filter')
        
        return tuple(filter(self.function, input))
    

class Apply(Module):
    """Apply given function to input Tensor
    """
    def __init__(self, function: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, input: Tensor):
        return self.function(input)
