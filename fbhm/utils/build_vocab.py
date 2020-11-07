'''
	Maan Qraitem 
'''

import sys
sys.path.append("../")
from vocab import Vocab 


def build_vocab(root_dir, filename):
	vocab = Vocab(root_dir)
	vocab.build_vocab("train.jsonl")
	vocab.init_glove("glove.6B.300d.txt")
	vocab.dumpfiles(filename)  
	

def main(): 
	root_dir = "/home/mqraitem/Documents/facebook_challenge" 
	build_vocab(root_dir, "vocab_data.pkl") 

if __name__ == "__main__": 
	main() 
