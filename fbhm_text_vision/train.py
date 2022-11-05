
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
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np 
from utils.logger import Logger 
from datetime import datetime
import os.path as osp
import pandas as pd

def evaluate(data_loader, model, device, mode, criterion=None): 
  
  loss = 0
  acc  = 0 

  sig_out = [] 
  truth_labels = []
  model_labels = [] 
  out_ids = [] 

  model.eval() 
  with torch.no_grad():
    sigmoid = nn.Sigmoid() 
    for sample in data_loader:  
      vision_feat = sample['img_feature'].to(device) 
      text_feat = sample['tokens'].to(device) 
      ids = sample['id'] 
      out = model(vision_feat, text_feat) 
      
      if mode in ['train', 'test']: 
        labels = sample['label'].to(device).view((-1,1)).float()
      
      out_sigmoid = sigmoid(out)
      pred_labels = out_sigmoid.clone() 
      pred_labels[pred_labels>0.5] = 1
      pred_labels[pred_labels<0.5] = 0 
   

      if mode in ['train', 'test']: 
        acc += torch.sum(pred_labels == labels).float()
        labels_list = labels.view(-1,).tolist()
        truth_labels.extend(labels_list)  

      if mode == 'train':
        loss_batch = criterion(out, labels)
        loss += loss_batch 
      
      out_sigmoid = out_sigmoid.view(-1,).tolist()  
      sig_out.extend(out_sigmoid)
      model_labels.extend(pred_labels.view(-1,).tolist()) 
      out_ids.extend(ids)

  model.train()

  all_arrays = { 
    'sig_out':sig_out,
    'truth_labels':truth_labels, 
    'model_labels':model_labels, 
    'ids':out_ids
  }
  
  stats = {} 
  if mode in ['train', 'test']: 
    loss = loss/len(data_loader.dataset)
    acc = acc/len(data_loader.dataset)
    aoc = roc_auc_score(np.array(truth_labels), np.array(sig_out), average='macro')  
     
    stats['loss'] = loss
    stats['acc'] = acc 
    stats['aoc'] = aoc

  return all_arrays, stats

def train(train_loader, valid_loader, model, epochs, model_dir, model_name, logger): 
  
  criterion = nn.BCEWithLogitsLoss()
  optim = torch.optim.Adam(model.parameters())
  scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=7, gamma=0.1)
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  model.to(device) 

  print("Using device %s"%device) 
  logger.write_model(model)
  best_vald_loss = -np.inf
  for epoch in range(epochs): 
    for _, sample in enumerate(tqdm(train_loader)):
      vision_feat = sample['img_feature'].to(device)
      text_feat = sample['tokens'].to(device) 
      label = sample['label'].to(device).view((-1, 1)).float()
      out = model(vision_feat, text_feat) 
      loss = criterion(out, label)
      loss.backward() 
      nn.utils.clip_grad_norm_(model.parameters(), 0.25)

      optim.step() 
      optim.zero_grad()
    
    scheduler.step()

    _, valid_stats = evaluate(valid_loader, model, device, 'train', criterion)
    _, train_stats = evaluate(train_loader, model, device, 'train', criterion) 
    stats = "Epoch %d | Train Loss %.6f, Train acc %.2f, Train aoc %.2f | Valid Loss %.6f Valid acc %.2f, Valid aoc %.2f"%(epoch, train_stats['loss'], train_stats['acc'], train_stats['aoc'], valid_stats['loss'], valid_stats['acc'], valid_stats['aoc'])
      
    logger.write_stats(stats) 
    print(stats)

    if valid_stats['aoc'] > best_vald_loss: 
      model_filename = "epoch_%d_valid_loss_%.4f_full"%(epoch, valid_stats['loss']) 
      torch.save(model, osp.join(model_dir, model_name, model_filename))
      print("Saved model: %s"% model_filename) 
      best_vald_loss = valid_stats['aoc']
   
  logger.close_file()
  return model, criterion, device

def main(): 

  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--build_vocab', type=int, default=1) 
  parser.add_argument('--batch_size', type=int, default=32) 
  parser.add_argument('--log_dir', type=str, default="../logs")
  parser.add_argument('--data_dir', type=str, default="../data") 
  parser.add_argument('--model_name', type=str, default="default")
  parser.add_argument('--model_dir', type=str, default="../runs")  
 
  args = parser.parse_args()

  dt_string = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
  logger = Logger(osp.join(args.log_dir, args.model_name, dt_string+".txt"))
  vocab = Vocab(args.data_dir)
 
  if args.build_vocab: 
    vocab.build_vocab("train.jsonl")
    vocab.build_vocab("dev_seen.jsonl") 
    vocab.build_vocab("test_seen.jsonl") 
    vocab.init_glove('glove.twitter.27B.200d.txt')
    vocab.dumpfiles("vocab.pkl")
  else: 
    vocab.loadfiles("vocab.pkl") 

  train_data = FbDataset('train', args.data_dir, vocab) 
  valid_data = FbDataset('dev_seen', args.data_dir, vocab)   
  test_data = FbDataset('test_seen', args.data_dir, vocab, mode='test')   

  train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=1) 
  valid_loader = DataLoader(valid_data, args.batch_size, num_workers=1)  
  test_loader = DataLoader(test_data, args.batch_size, num_workers=1) 

  num_hidden = 500
  model = build_main_model(num_hidden, len(vocab), vocab, dropout=0.5, text_dim=200, vision_dim = 2048)
  
  model, criterion, device = train(train_loader, valid_loader, model, args.epochs, args.model_dir, args.model_name, logger)  
 
if __name__ == "__main__": 
  main() 
