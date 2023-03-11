import torch
from torch import nn

from pytorch_toolz.itertools import Accumulate, Chain, Repeat, Slice
from .framework import TorchTest


class TestItertools(TorchTest):
    def setUp(self) -> None:
        self.pipeline = nn.Sequential(
            Repeat(2),
            Repeat(2),
            Chain(),
            Slice(-2),
            Accumulate(torch.add),
        )

    def test_itertools(self):
        tensor = torch.randn((5, 5))
        res1, res2 = self.pipeline(tensor)
        self.assertEqual(tensor, res1)
        self.assertEqual(tensor, res2)
