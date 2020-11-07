
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

def evaluate(data_loader, model, criterion, device): 
  
  loss = 0
  acc  = 0 

  all_preds = [] 
  all_labels = [] 
  model.eval() 
  with torch.no_grad():
    sigmoid = nn.Sigmoid() 
    for sample in data_loader:  
      vision_feat = sample['img_feature'].to(device) 
      text_feat = sample['tokens'].to(device) 
      label = sample['label'].to(device).view((-1,1)).float()
      out = model(vision_feat, text_feat) 
  
      loss_batch = criterion(out, label)
      loss += loss_batch 
    
      out_sigmoid = sigmoid(out)
      pred_labels = out_sigmoid.clone() 
      pred_labels[pred_labels>0.5] = 1
      pred_labels[pred_labels<0.5] = 0 

      acc += torch.sum(pred_labels == label).float()
    
      out_sigmoid = out_sigmoid.view(-1,).tolist()  
      label = label.view(-1,).tolist()
  
      all_preds.extend(out_sigmoid)
      all_labels.extend(label)  
    
  model.train()
  loss = loss/len(data_loader.dataset)
  acc = acc/len(data_loader.dataset)
  aoc = roc_auc_score(np.array(all_labels), np.array(all_preds), average='macro')  
  return loss, acc, aoc 

def train(train_loader, valid_loader, model, epochs, model_dir, model_name, logger): 
  
  criterion = nn.BCEWithLogitsLoss()
  optim = torch.optim.Adam(model.parameters())
  best_valid_score = 0 
 
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
  model.to(device) 

  print("Using device %s"%device) 
  logger.write_model(model)
  best_vald_loss = np.inf
  for epoch in range(epochs): 
    for _, sample in enumerate(tqdm(train_loader)):
      vision_feat = sample['img_feature'].to(device)
      text_feat = sample['tokens'].to(device) 
      label = sample['label'].to(device).view((-1, 1)).float()
    
      out = model(vision_feat, text_feat) 
      loss = criterion(out, label)
      loss.backward() 

      optim.step() 
      optim.zero_grad()

    valid_loss, valid_acc, valid_aoc = evaluate(valid_loader, model, criterion, device)
    train_loss, train_acc, train_aoc = evaluate(train_loader, model, criterion, device) 
    stats = "Epoch %d | Train Loss %.6f, Train acc %.2f, Train aoc %.2f | Valid Loss %.6f Valid acc %.2f, Valid aoc %.2f"%(epoch, train_loss, train_acc, train_aoc, valid_loss, valid_acc, valid_aoc)
      
    logger.write_stats(stats) 
    print(stats)

    if valid_loss < best_vald_loss: 
      model_filename = "epoch_%d_valid_loss_%.4f"%(epoch, valid_loss) 
      torch.save(model.state_dict(), osp.join(model_dir, model_name, model_filename))
      print("Saved model: %s"% model_filename) 
      best_vald_loss = valid_loss

  logger.close_file()

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
    vocab.build_vocab("dev_unseen.jsonl") 
    vocab.init_glove('glove.twitter.27B.25d.txt')
    vocab.dumpfiles("vocab.pkl")
  else: 
    vocab.loadfiles("vocab.pkl") 

  train_data = FbDataset('train', args.data_dir, vocab) 
  valid_data = FbDataset('dev_unseen', args.data_dir, vocab)   

  train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=1) 
  valid_loader = DataLoader(valid_data, args.batch_size, num_workers=1) 
  
  num_hidden = 100
  model = build_main_model(num_hidden, len(vocab), vocab, dropout=0.4)
  
  train(train_loader, valid_loader, model, args.epochs, args.model_dir, args.model_name, logger)  
 
main() 
