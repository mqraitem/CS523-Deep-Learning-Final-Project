'''
  * Maan Qraitem
'''



import torch 
import torch.nn as nn 
from fc_net import FCNet


class VisionModel(nn.Module): 
  def __init__(self, vision_dim, num_hidden): 
    super(VisionModel, self).__init__()
    self.fc_layers = FCNet([vision_dim, num_hidden, num_hidden], 0.2) 
 
  def forward(self, x): 
    n = torch.norm(x, p=2, dim=-1).detach()
    x = x.div(xn.unsqueeze(dim=-1))
    return x
    #return self.fc_layers(x) 

  
