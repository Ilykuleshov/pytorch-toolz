from typing import Tuple, Union, TypeVar, Generic, Callable
from torch.nn import Module
from torch import Tensor, NumberType
from operator import itemgetter

class Apply(Module):
    """Apply given function to input
    """
    def __init__(self, function) -> None:
        super().__init__()
        self.function = function

    def forward(self, *input):
        return self.function(*input)


T = TypeVar('T')
class ItemGetter(Module):
    """Get specified items from input tuple
    """
    def __init__(self, *items: int) -> None:
        super().__init__()
        self.function = itemgetter(*items)
    
    def forward(self, input: Tuple[T, ...]) -> Union[T, Tuple[T]]:
        return self.function(input)
