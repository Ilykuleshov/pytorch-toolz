from typing import Callable, Tuple, Union, overload
from itertools import accumulate, repeat, chain
from torch.nn import Module


class Accumulate(Module):
    """Accumulate input using function, see itertools.accumulate
    """
    @overload
    def __init__(self, function: Callable, /) -> None: ...
    @overload
    def __init__(self, function: Callable, *, initial) -> None: ...
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, input: Tuple):
        return tuple(accumulate(input, *self.args, **self.kwargs))


class Repeat(Module):
    """Repeat input r times, output a tuple
    """
    def __init__(self, r: int) -> None:
        super().__init__()
        self.r = r

    def forward(self, input):
        return tuple(repeat(input, self.r))


class Slice(Module):
    """Get a certain slice from input
    """
    @overload
    def __init__(self, __stop: int) -> None: ...
    @overload
    def __init__(self, __start: Union[int, None], __stop: Union[int, None], __step: Union[int, None]=...) -> None: ...
    
    def __init__(self, *args: Union[int, None]) -> None:
        super().__init__()
        self.slice = slice(*args)
        
    def forward(self, input: Tuple):
        return input[self.slice]


class Chain(Module):
    """Chain input tuples
    """
    def forward(self, input: Tuple[Tuple, ...]):
        return tuple(chain(*input))
