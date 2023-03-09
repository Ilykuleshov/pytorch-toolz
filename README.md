# pytorch-toolz
## Pitch
Building models in pytorch is, in essence, aligned with the functional paradigm: to build a model one defines a pipeline of functions that the inputs pass through to generate the output. And yet pytorch lacks a few essential functional tools that would allow to define such pipelines, which even raw python supports in functools (despite not originally being a functional programming language), such as `reduce`, `map`, `filter`. Out of the box, pytorch only supports function composition (`nn.Sequential`). This library aims to mitigate this issue by adding a couple of tools which, in my opinion, should be present in pytorch. 

### Readability
This greatly broadens the spectre of modules which can be built in a one-line fashion. While large, highly complicated modules probably shouldn't be defined this way (as readablity would suffer greatly), smaller modules (like resblock) will likely win in readability. Functional-style definition allows the user to encapsulate popular data-flow patterns into recognisable blocks.

### File clutter
The OOP approach inevitably leads to big project volumes, since classes are usually defined in separate files. One would normally not define a class inside a function, but would instead define it separately, and then import it. This is due to large code overhead for classes. The functional approach allows to substitute large cluttery classes with function calls! Of course, if a module is to be used repeatedly in a project, such code overhead is okay, but creating a file for a class, which will only be used once is overkill. This is especially the case when working in standalone Jupyter notebooks.

### Building Block diversity
This wouldn't be a problem if there had been some universal high-quality library with commonly used pytorch modules, but that is not the case, and for good reason. There is, for example, no universal ResBlock everyone would be content with, so most CV projects define their own, in their own separate `blocks.py` file. Since some blocks are so subject to change, wouldn't it be more logical to define them on-the-fly? No agreed-on, universal ResBlock and ConvBlock structure has greater consequences: there is, for example, no universal UNet, despite the fact that the UNet architecture is generally the same across most projects: they all vary in the little blocks they use (so we end up with BottleneckUNet, ResUNet, 3dUnet, etc.)! What if, in addition to this, we had a library with some common architecture patterns (UNet, RNN, etc.), which would simply arrange the given building blocks (whatever they may be, one can build them on-the-fly) in a predefined structure? They would be much more reusable, maybe not to the point of being added as standard to pytorch, but at least to the point of covering most use cases.

### Coding freedom
I may be overestimating the impact of this approach, but in the very least it would give pytorch users much greater freedom over model building. After all, there should be *many ways to skin a python*.

## Philosophy
This library is intended to be tiny. It doesn't need to contain anything except the standard tools, already present in python functional modules. This way, it will stay encapsulated, compact and consise, a small toolbox for a big number of things. If some standard itertool/functool is not present here and you find a use for it, please submit an issue/pull-request.

## Examples
### ResBlock
**functional way**:
```
def conv(in_ch, out_ch):
    return nn.Sequential(
        nn.BatchNorm3d(in_ch),
        nn.ReLU(),
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding='same')
    )

def resblock(in_ch, out_ch):
    assert in_ch == out_ch, (in_ch, out_ch)
    hidden_ch = in_ch // 4
    return nn.Sequential(
        Parallel(
            nn.Identity(),
            nn.Sequential(
                conv(in_ch, hidden_ch),
                conv(hidden_ch, out_ch)
            )
        ),
        Reduce(torch.add)
    )
```

**OOP:**
`blocks.py`
```
class ConvBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
    self.block = nn.Sequential(
        nn.BatchNorm3d(in_ch),
        nn.ReLU(),
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding='same')
    )
   
  def forward(self, x):
    return self.block(x)
    
   
class ResBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
    assert in_ch == out_ch
    hidden_ch = in_ch // 4
    self.in_block = ConvBlock(in_ch, hidden_ch)
    self.out_block = ConvBlock(hidden_ch, out_ch)
    
  def forward(self, x):
    y = self.out_block(self.in_block(x))
    return x + y
```
