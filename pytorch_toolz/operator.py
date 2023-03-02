from torch.nn import Module

class Add(Module):
    @staticmethod
    def forward(a, b):
        return a + b


class Sub(Module):
    @staticmethod
    def forward(a, b):
        return a - b
