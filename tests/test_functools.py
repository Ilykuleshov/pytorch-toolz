from pytorch_toolz.functools import Filter, Map, Reduce, Parallel, Sequential
from pytorch_toolz.operator import Apply
from torch.nn import Identity
import torch

from .framework import TorchTest

class TestFunctools(TorchTest):
    def setUp(self) -> None:
        self.pipeline = Sequential(
            Parallel(Apply(torch.neg), Identity()),
            Parallel(Reduce(torch.add), Reduce(torch.mul), Reduce(torch.sub), Reduce(torch.div)),
            Parallel(
                Reduce(torch.add), 
                Sequential(Map(lambda x: x * 2), Reduce(torch.add)), 
                Filter(lambda x: bool((x > 5).all().item())), 
            )
        )


    def test_parallel(self):
        tensor = torch.randn((5, 5))
        reduced, mapped, filtered = self.pipeline(tensor)

        intermediate = [-tensor + tensor, - tensor * tensor, -tensor - tensor, -tensor / tensor]

        self.assertEqual(reduced, sum(intermediate))
        self.assertEqual(mapped, sum(map(lambda x: 2 * x, intermediate)))
        self.assertEqual(list(filtered), list(filter(lambda x: bool((x > 5).all().item()), intermediate)))
