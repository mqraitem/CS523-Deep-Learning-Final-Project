'''
  * Maan Qraitem
'''



import torch 
import torch.nn as nn 
from fc_net import FCNet


class VisionModel(nn.Module): 
  def __init__(self, vision_dim, num_hidden): 
    super(VisionModel, self).__init__()
    self.fc_layers = FCNet([vision_dim, num_hidden, num_hidden], nn.ReLU(), 0.2) 
 
  def forward(self, x): 
    return x
    #return self.fc_layers(x) 

  
