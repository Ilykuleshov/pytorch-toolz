from torch import nn

class Parallel(nn.Sequential):
    def forward(self, input):
        output = []
        for module in self:
            if isinstance(input, tuple):
                output.append(module(*input))
            else:
                output.append(module(input))
                
        return tuple(output)
