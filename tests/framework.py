from typing import Iterable
import unittest
import torch
from itertools import starmap


class TorchTest(unittest.TestCase):
    def assertEqual(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(isinstance(b, torch.Tensor))
            self.assertTrue((a == b).all)
            return
        elif isinstance(a, Iterable):
            return all(starmap(self.assertEqual, zip(a, b)))
