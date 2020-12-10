
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
import pandas as pd
from train import evaluate

def generate_test_file(data_loader, model, device):
  all_arrays, _ = evaluate(data_loader, model, device, 'comp')
  all_arrays['sig_out'] = ['%.1f'%x for x in all_arrays['sig_out']]
  all_arrays['model_labels'] = [int(x) for x in all_arrays['model_labels']]
  d = {'id':all_arrays['ids'],
       'proba':all_arrays['sig_out'],
       'label':all_arrays['model_labels']}

  df = pd.DataFrame(data=d)
  df.set_index('id', inplace=True)
  df.to_csv('test.csv')


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../data')
  parser.add_argument('--model_path', type=str, default='../runs')
  parser.add_argument('--batch_size', type=int, default=32)

  args = parser.parse_args()

  vocab = Vocab(args.data_dir)
  vocab.loadfiles('vocab.pkl')

  test_data = FbDataset('test_seen', args.data_dir, vocab, mode='comp')
  test_loader = DataLoader(test_data, args.batch_size, num_workers=1)

  valid_data = FbDataset('dev_seen', args.data_dir, vocab, mode='train')
  valid_loader = DataLoader(valid_data, args.batch_size, num_workers=1)

  device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
  model = torch.load(args.model_path, map_location=device)
  model.to(device)

  generate_test_file(test_loader, model, device)
  _, stats = evaluate(valid_loader, model, device, 'test')

  print("Validation Acc %.4f, Validation AUROC %.4f"%(stats['acc'], stats['aoc']))


if __name__ == "__main__":
  main()
