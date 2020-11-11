'''
  * Maan Qraitem 
  * Attention
'''

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import numpy as np

class Attention(nn.Module):
  def __init__(self, query_dim, context_dim, num_hidden):
    super(Attention, self).__init__() 
    self.query_dim = query_dim 
    self.context_dim = context_dim 
    self.num_hidden = num_hidden 

    self.linear1 = nn.Linear(query_dim + context_dim, num_hidden) 
    self.linear2 = nn.Linear(num_hidden, 1) 

    self.relu = torch.nn.ReLU() 
    self.softmax = torch.nn.Softmax(dim=1) 

  def forward(self, query, context, mask=None): 
    '''   
      query shape   = [batch_size, K, query_dim] 
      context shape = [batch_size,  context_dim] 
    '''
    K = query.size(1)
    context = context.unsqueeze(1).repeat(1, K, 1) 
    cq = torch.cat((query, context), 2)
    cq = self.linear1(cq)
    cq = self.relu(cq) 
    attention_weights = self.linear2(cq)
    if mask is not None: 
      attention_weights[mask] = -np.inf
    
    attention_weights = self.softmax(attention_weights)     
    return attention_weights
    

def main(): 
  attention = Attention(2, 2, 3) 
  query = torch.randn(10, 4, 2) 
  context = torch.randn(10, 2) 
  
  attention(query, context)

if __name__ == "__main__":
  main() 
    
    
