from typing import TypeVar, Tuple, Generic
from typing import Callable
from torch.nn import Module, Sequential
from torch import Tensor
from functools import reduce


class Parallel(Sequential):
    def forward(self, input):
        return tuple(module(input) for module in self)


class Reduce(Module):
    def __init__(self, function: Callable[[Tensor, Tensor], Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, input: Tuple[Tensor, ...]) -> Tensor:
        if not isinstance(input, tuple):
            raise RuntimeError('Non-tuple argument to Reduce')
        
        return reduce(self.function, input)


class Map(Module):
    def __init__(self, function: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, input: Tuple[Tensor, ...]) -> Tuple[Tensor]:
        return tuple(map(self.function, input))


class Filter(Module):
    def __init__(self, function: Callable[[Tensor], bool]) -> None:
        super().__init__()
        self.function = function

    def forward(self, input: Tuple[Tensor, ...]):
        return tuple(filter(self.function, input))
    

class Apply(Module):
    def __init__(self, function: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, input: Tensor):
        return self.function(input)
