from typing import TypeVar
from typing import Callable
from torch.nn import Module
from functools import reduce

T = TypeVar('T')
U = TypeVar('U')

class Reduce(Module):
    def __init__(self, function: Callable[[T, T], T]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *args: T) -> T:
        return reduce(self.function, args)


class Map(Module):
    def __init__(self, function: Callable[[T], U]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *args):
        return tuple(map(self.function, args))
    

class Filter(Module):
    def __init__(self, function: Callable[[T], bool]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *args):
        return tuple(filter(self.function, args))
