
'''
  * Maan Qraitem 
  * Attention
'''

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from main_model import build_main_model 
import argparse 
from vocab import Vocab
from dataset import FbDataset
from torch.utils.data import DataLoader


def evalulate(data_loader, model): 
  
  loss = 0
  for sample in data_loader:  
    vision_feat = sample['img_feature']
    text_feat = sample['tokens']
    label = sample['label']       
    out = model(vision_feat, text_feat) 
    loss_batch = criterion(out, label)
    loss += loss_batch 
    
  loss = loss/len(dataloader.dataset)
  return loss 

def train(train_loader, valid_loader, model, epochs): 
  
  criterion = nn.BCEWithLogitsLoss()
  optim = torch.optim.Adam(model.parameters())
  best_valid_score = 0 

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
  model.to(device) 

  for epoch in range(epochs): 
    for sample in train_loader:
      vision_feat = sample['img_feature'].to(device)
      text_feat = sample['tokens'].to(device) 
      label = sample['label'].top(device)

      out = model(vision_feat, text_feat) 
      loss = criterion(out, label)
      loss.backward() 

      optim.step() 
      optim.zero_grad()

    
    print("Epoch %d, Loss %.3f"%(epoch, evalulate(valid_loader, model))  
       

def main(): 

  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--build_vocab', type=int, default=1) 
  parser.add_argument('--batch_size', type=int, default=128) 
  parser.add_argument('--log_dir', type=str, default="../logs")
  parser.add_argument('--data_dir', type=str, default="../data") 
  args = parser.parse_args()

  
  vocab = Vocab(args.data_dir) 
  if args.build_vocab: 
    vocab.build_vocab("train.jsonl")
    vocab.build_vocab("dev_unseen.jsonl") 
    vocab.dumpfiles("vocab.pkl")
  else: 
    vocab.loadfiles("vocab.pkl") 

  #train_data = FbDataset('train.jsonl', args.data_dir, vocab) 
  valid_data = FbDataset('dev_unseen.jsonl', args.data_dir, vocab)   

  #train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=1) 
  valid_loader = DataLoader(valid_data, args.batch_size, num_workers=1) 
  
  num_hidden = 100
  model = build_main_model(num_hidden, len(vocab.word2idx))
  
  train(valid_loader, valid_loader, model, args.epochs)  
 
main() 
