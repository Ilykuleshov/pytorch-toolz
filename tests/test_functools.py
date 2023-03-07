from typing import Union, Iterable
from itertools import starmap
import unittest
from pytorch_toolz.nn import Filter, Map, Reduce, Apply, Parallel
from torch.nn import Sequential, Identity
import torch
import random


class TestFunctools(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = Sequential(
            Parallel(Apply(lambda x: -x), Identity()),
            Parallel(Reduce(torch.add), Reduce(torch.mul), Reduce(torch.sub), Reduce(torch.div)),
            Parallel(Reduce(torch.add), Sequential(Map(lambda x: x * 2), Reduce(torch.add)), Filter(lambda x: bool((x > 5).all().item())))
        )

    def assertEqual(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(isinstance(b, torch.Tensor))
            self.assertTrue((a == b).all)
            return
        elif isinstance(a, Iterable):
            return all(starmap(self.assertEqual, zip(a, b)))

    def test_parallel(self):
        tensor = torch.randn((5, 5))
        reduced, mapped, filtered = self.pipeline(tensor)

        intermediate = [-tensor + tensor, - tensor * tensor, -tensor - tensor, -tensor / tensor]

        self.assertEqual(reduced, sum(intermediate))
        self.assertEqual(mapped, sum(map(lambda x: 2 * x, intermediate)))
        self.assertEqual(list(filtered), list(filter(lambda x: bool((x > 5).all().item()), intermediate)))
