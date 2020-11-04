'''
  * Maan Qraitem
  * Language model 
'''

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from attention import Attention

class LanguageModel(nn.Module): 
  def __init__(self, num_tokens, token_dim, num_hidden): 
    super(LanguageModel, self).__init__() 
    self.embed = nn.Embedding(num_tokens, token_dim)
    self.attention = Attention(token_dim, token_dim, num_hidden) 

  def forward(self, words_feat): 
    words_rep = self.embed(words_feat)
    context = torch.mean(words_rep, dim = 1)
    attention_weights = self.attention(words_rep, context)
    words_rep = (attention_weights * words_rep).sum(1) 
    return words_rep   


def main(): 
  lm = LanguageModel(10, 5, 3) 
  example = torch.tensor([[1, 2, 1], [2,2,2]]) 
 
  lm(example)

if __name__ == "__main__":
  main() 
    
    
   
