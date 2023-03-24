from typing import TypeVar, Tuple, Generic, Callable, overload, Union
from torch.nn import Module, Sequential as Sequential_
from torch.nn.parameter import Parameter
from torch import Tensor
from functools import reduce


class Sequential(Sequential_):
    """A sequential container, nn.Sequential with control over unpacking.
    """

    def __init__(self, *args, unpack=False):
        super().__init__(*args)
        self.unpack = unpack

    def forward(self, *input):
        if not self.unpack:
            input, = input

        for module in self:
            if self.unpack:
                input = module(*input)
            else:
                input = module(input)

        return input


class Parallel(Sequential):
    """A parallel container simmilar to toolz.juxt, opposite of Map.
    Applies contained layers to input, returns tuple of results. Usage same as Sequential.
    """

    def forward(self, *input):
        if not self.unpack:
            input, = input

        results = []
        for module in self:
            if self.unpack:
                results.append(module(*input))
            else:
                results.append(module(input))

        return tuple(results)


class Reduce(Module):
    """A module, simmilar to functools.reduce. Applies a two-argument function to given inputs.
    Example usage, simplified residual block:
    ```
    res_block = nn.Sequential(
        Parallel(nn.Conv2d(...), nn.Identity()),
        Reduce(torch.add)
    )
    ```
    """
    @overload
    def __init__(self, function: Callable, /) -> None: ...
    @overload
    def __init__(self, function: Callable, initial, /) -> None: ...

    def __init__(self, *args) -> None:
        super().__init__()
        self.function = args[0]
        if len(args) == 2:
            self.initial = args[1]
        
        if len(args) > 2:
            raise TypeError(f"Invalid number of arguments passed to {self.__class__.__name__}")

    def forward(self, input: Tuple):
        if hasattr(self, 'initial'):
            return reduce(self.function, input, self.initial)
        
        return reduce(self.function, input)


class _Functool(Module):
    def __init__(self, function: Callable) -> None:
        super().__init__()
        self.function = function


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
