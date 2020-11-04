'''
	Maan Qraitem
	Hateful meme classification 
'''

import numpy as np 
from nltk.tokenize import word_tokenize
import pandas as pd
import os.path as osp
from tqdm import tqdm 
import pickle

class Vocab(): 
	
	def __init__(self, root_dir):
		self.root_dir = root_dir 
		self.word2idx = {} 
		self.wordmatrix = None 
		self.vocab_size = 0

	def padding_word_idx(self): 
		return len(self.word2idx.keys()) 

	def tokenize(self, meme_text):
			#TODO: improve word filtering. 
			meme_text = meme_text.lower()
			meme_text = meme_text.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('!', '') 
			meme_tokens = word_tokenize(meme_text) 
			return meme_tokens 

	def build_vocab(self, datafile_name):
		meme_data = pd.read_json(osp.join(self.root_dir, datafile_name), lines=True)
		print("Building Vocabulary")
		for index, meme_entry in tqdm(meme_data.iterrows(), total=meme_data.shape[0]):	
			meme_tokens = self.tokenize(meme_entry['text']) 
			for token in meme_tokens: 
				if token in self.word2idx: 
					continue
				else: 
					self.word2idx[token] = self.vocab_size 
					self.vocab_size += 1

	def init_glove(self, datafile_name): 
		print("Building Embedding matrix")
		with open(osp.join(self.root_dir, datafile_name), 'r') as f:
			entries = f.readlines()
		
		embedding_dim = len(entries[0].split(' ')) - 1
		print('embedding dim: %d' % embedding_dim)
		self.wordmatrix = np.zeros((len(self.word2idx), embedding_dim), dtype=np.float32)	
		
		for entry in tqdm(entries): 
			word = entry.split(" ")[0] 
			emb  = list(map(float, entry.split(" ")[1:]))
			emb  = np.array(emb)
			if word in self.word2idx:
				idx = self.word2idx[word] 
				self.wordmatrix[idx] = emb	

	def dumpfiles(self, filename):
		path = osp.join(self.root_dir, filename)  
		pickle.dump([self.word2idx, self.wordmatrix], open(path, 'wb'))
		print("word2idx, wordmatrix dumped to %s "%path) 


	def loadfiles(self, filename): 
		assert osp.isfile(osp.join(self.root_dir, filename)),"%s does not exist"%filename
		path = osp.join(self.root_dir, filename) 
		self.word2idx, self.wordmatrix = pickle.load(open(path, 'rb'))
		return self.word2idx, self.wordmatrix 		

