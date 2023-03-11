import torch
from torch import nn

from pytorch_toolz.operator import Apply, ItemGetter
from .framework import TorchTest


class TestOperator(TorchTest):
    def setUp(self) -> None:
        self.pipeline = nn.Sequential(
            Apply(lambda x: tuple(x * i for  i in range(10))),
            ItemGetter(0, 2, 4, 6)
        )

    def test_operator(self):
        tensor = torch.randn((5, 5))
        zero, two, four, six = self.pipeline(tensor)
        self.assertEqual(zero, tensor * 0)
        self.assertEqual(two, tensor * 2)
        self.assertEqual(four, tensor * 4)
        self.assertEqual(six, tensor * 6)
