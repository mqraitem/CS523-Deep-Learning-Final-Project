import h5py
import numpy as np 
import os.path as osp 
import pickle 
import argparse
import pandas as pd 
from tqdm import tqdm

def main(): 
  
  parser = argparse.ArgumentParser() 
  parser.add_argument('--data_dir', type=str, default='../data') 
  parser.add_argument('--dataset', type=str, default='train') 
  args = parser.parse_args() 
  img_features = []   
  img2idx = {} 

  id_data = pd.read_json(osp.join(args.data_dir, args.dataset+".jsonl"), lines=True) 

  for index, meme_entry in tqdm(id_data.iterrows(), total=id_data.shape[0]): 
    
    meme_id = meme_entry['img'].split('.')[0].split('/')[1] 
    img_feature_name = meme_id + ".npz"
    img_feature = np.load(osp.join(args.data_dir, 'img_features', img_feature_name)) 
    img_features.append(img_feature['x'])  
    img2idx[meme_id] = index  

  img_features = np.array(img_features) 

  np.save(osp.join(args.data_dir, 'img_features_%s'%args.dataset), img_features)
  print("img_features size: ", img_features.shape)
  print(img_features[0])
  with open(osp.join(args.data_dir, 'img2idx_%s.pkl'%args.dataset), 'wb') as handle:
    pickle.dump(img2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__': 
  main()
