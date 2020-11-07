from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc_net import FCNet  
from language_model import LanguageModel
from attention import Attention 
from vision_model import VisionModel 

class MainModel(nn.Module): 
  def __init__(self, language_model, vision_model, vision_dim, token_dim, num_hidden, dropout):  
    super(MainModel, self).__init__() 
    self.language_model = language_model
    self.vision_model = vision_model 
    self.attention = Attention(vision_dim, token_dim, num_hidden)

    self.vision_fcnet = FCNet([vision_dim, num_hidden], nn.ReLU(), 0.4)
    self.text_fcnet = FCNet([token_dim, num_hidden], nn.ReLU(), 0.4) 

    self.classifier_layers = [
      weight_norm(nn.Linear(num_hidden, num_hidden),dim=None),
      nn.ReLU(),  
      nn.Dropout(dropout, inplace=True),
      weight_norm(nn.Linear(num_hidden, 1),dim=None),
    ] 

    self.classifier = nn.Sequential(*self.classifier_layers)

  def forward(self, vision_feat, words_feat): 
    '''
      text_input = [batches, text_feature_size] 
      vision_input = [batches, num_objects, vision_feature_size] 
    '''
       
    meme_feat = self.language_model(words_feat) 
    vision_feat = self.vision_model(vision_feat)
  
    attention_weights = self.attention(vision_feat, meme_feat) #[batches, num_obj] 
    vision_feat = (attention_weights * vision_feat).sum(1)
    
    vision_feat = self.vision_fcnet(vision_feat) 
    meme_feat = self.text_fcnet(meme_feat) 

    combined_feat = vision_feat * meme_feat
    #combined_feat = vision_feat
    out = self.classifier(combined_feat)  
      
    return out 


def build_main_model(num_hidden, num_tokens, vocab, dropout):
  language_model = LanguageModel(num_tokens, 25, num_hidden)
  vision_model = VisionModel(2048, 2048) 
  language_model.init_embed(vocab.wordmatrix)
  main_model = MainModel(language_model, vision_model, 2048, 25, num_hidden, dropout) 
  
  return main_model


def main(): 
  num_tokens = 10
  token_dim = 4 
  num_hidden = 3
  vision_dim = 10 
  language_model = LanguageModel(num_tokens, token_dim, num_hidden)
  main_model = MainModel(language_model, vision_dim, token_dim, num_hidden) 
  
  vision_example = torch.randn(2, 3, vision_dim)
  text_example = torch.tensor([[1, 1, 2, 2], [3, 4, 5, 6]])

  main_model(vision_example, text_example)
   


if __name__ == "__main__":
  main() 
