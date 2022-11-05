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
    self.embed = nn.Embedding(num_tokens, token_dim, padding_idx=0)
    self.attention = Attention(token_dim, token_dim, num_hidden) 
    self.num_tokens = num_tokens 
    self.token_dim = token_dim 

  def init_embed(self, loaded_embed): 
    loaded_embed = torch.from_numpy(loaded_embed)
    self.embed.weight.data = loaded_embed
    self.embed.weight.requires_grad=False
    print("Embedding initialized")

  def forward(self, words_feat): 
    mask = words_feat == 0 
    words_rep = self.embed(words_feat)
    context = torch.mean(words_rep, dim = 1)
    attention_weights = self.attention(words_rep, context, mask)
    words_rep = (attention_weights * words_rep).sum(1)
    return words_rep   


def main(): 
  lm = LanguageModel(10, 5, 3) 
  example = torch.tensor([[1, 2, 1], [2,2,2]]) 
 
  lm(example)

if __name__ == "__main__":
  main() 
    
    
   
