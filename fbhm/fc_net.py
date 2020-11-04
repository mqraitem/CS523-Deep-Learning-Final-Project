from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch

class FCNet(nn.Module):
  def __init__(self, layer_sizes, nonlinearity):
    super(FCNet, self).__init__()
    layers = []
    for i in range(len(layer_sizes)-2):
      in_dim = layer_sizes[i]
      out_dim = layer_sizes[i+1]
      layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
      layers.append(nonlinearity)
    
    layers.append(weight_norm(nn.Linear(layer_sizes[-2], layer_sizes[-1]), dim=None))
    layers.append(nonlinearity)

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    return self.main(x)


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10], nn.ReLU())
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20], nn.ReLU())
    print(fc2)

    example = torch.randn([2, 10]) 
    fc1(example)
