'''
  Maan Qraitem  
  Hateful Meme Classification 
'''

from __future__ import print_function, division
import os
import os.path as osp 
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json 
from vocab import Vocab
from tqdm import tqdm
import pickle
#import cv2

class FbDataset(Dataset): 
  def __init__(self, dataset, root_dir, vocab, mode='train'):
    self.root_dir = root_dir
    self.meme_data = pd.read_json(osp.join(root_dir, dataset+'.jsonl'), lines=True) 
    self.img_features = np.load(osp.join(root_dir, 'img_features_%s.npy'%dataset))
    self.mode = mode 
    with open(osp.join(root_dir, 'img2idx_%s.pkl'%dataset), 'rb') as handle:
      self.img2idx = pickle.load(handle)
    
    self.entries = [] 
    self.vocab = vocab
    self.max_length = 15
     

    self.load_data() 
    self.tokenize() 

  def add_entry(self, meme_entry):
    image_id =  meme_entry['img'].split(".")[0].split("/")[1]
    meme_text = meme_entry['text'] 
    meme_img_feature = self.img_features[self.img2idx[image_id]]  
    meme_id = meme_entry['id'] 

    entry = { 
        'id':meme_id,
        'imag_id': image_id,
        'text': meme_text, 
        'img_feature': meme_img_feature
      }
    
    if self.mode == 'train': 
      meme_label = np.array(meme_entry['label']).astype(np.float)
      entry['label'] = meme_label 

    self.entries.append(entry)

  def load_data(self): 
    for index, meme_entry in tqdm(self.meme_data.iterrows(), total=self.meme_data.shape[0]):
      self.add_entry(meme_entry)  

  
  def tokenize(self): 
    for entry in self.entries: 
      meme_text = entry['text']
      meme_tokens = self.vocab.tokenize(meme_text)
      meme_tokens = [self.vocab.word2idx[token] for token in meme_tokens] 
      meme_tokens = meme_tokens[:self.max_length]  
      if len(meme_tokens) < self.max_length: 
        padding = [self.vocab.padding_word_idx()] * (self.max_length - len(meme_tokens)) 
        meme_tokens = meme_tokens + padding
        
      assert len(meme_tokens) == self.max_length, "meme text size is not %d"%self.max_length
      entry['text_tokens'] = np.array(meme_tokens) 


  def __len__(self): 
    return len(self.entries) 

  def __getitem__(self, index):
    entry = self.entries[index] 
    entry_id = str(entry['id'])
    entry_tokens = entry['text_tokens']
    entry_text   = entry['text']  
    entry_img_feature = entry['img_feature']    
  
    sample = { 
      'id'    : entry_id, 
      'tokens': entry_tokens, 
      'text'  : entry_text, 
      'img_feature' : entry_img_feature, 
    } 

    if self.mode == 'train': 
      entry_label  = entry['label'] 
      sample['label'] = entry_label 

    return sample 

def main():
  root_path = '../../data'
  vocab = Vocab(root_path) 
  vocab.loadfiles('vocab_data.pkl') 
  fbdataset = FbDataset('train.jsonl', root_path, vocab) 
  for i in range(len(fbdataset)): 
    sample = fbdataset[i]
    #print(sample['id'])
    #print(sample['tokens']) 
    #print(sample['text']) 
    #cv2.imshow('image',sample['img'])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()  

if __name__ == "__main__":
  main()
  

