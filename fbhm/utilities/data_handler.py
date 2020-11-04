import matplotlib
import pandas as pd
from pathlib import Path
from matplotlib import pyplot

class DataHandler:
    def __init__(self, dfp):
        super().__init__()
        self.dfp = dfp # assumes dfp to be a Path object
        self.da = {}
    
    def load_all_data(self):
        """ Assumes jsonl file inside the data directory
            Input: DataHandler Class object, No parameters
            Output: (Pandas data frame) data frame containing folowing cols:
                    'id', 'img', 'text'.
                    The data frame contains all the unique
                    entries in all the .jsonl files irrespective of train and
                    dev and test
        """
        # Find all '.jsonl' files in data directory
        jsonl_files = sorted(Path(self.dfp).rglob('**/*.jsonl'))
        if jsonl_files is None:
            print("Data directory contains no '.jsonl' formatted files")
            raise
        
        # concatenate all data frames read from all json line files in data 
        data = pd.concat([pd.read_json(f, lines=True) for f in jsonl_files])

        # remove all duplicate entries and reindex the dataset
        data = data.drop_duplicates().reset_index(drop=True)
        unlabeled_data = data.loc[data['label'].isna()].reset_index(drop=True)
        labeled_data = data.loc[data['label'].notna()].reset_index(drop=True)
        return data, labeled_data, unlabeled_data

    def compute_data_analytics(self, df):
        # Class and Shape analysis
        pd.set_option("display.precision", 2)
        self.da['SHAPE'] = df.shape
        #self.da['CA'] = df['label'].value_counts(normalize=True)

        # Text analysis
        print(df.shape)
        class0 = df.loc[df['label'] == 0.0]
        class1 = df.loc[df['label'] == 1.0]
        print(f'# Class 0 data points: {class0.shape}')
        print(f'# Class 1 data points: {class1.shape}')

        text_df = class0['text'].apply(lambda s: len(s))
        pyplot.hist(text_df, bins='auto', alpha=0.75, ec='black', label='non-hateful')
        self.da['TA0'] = text_df.describe()

        text_df = class1['text'].apply(lambda s: len(s))
        pyplot.hist(text_df, bins='auto', alpha=0.5, ec='black', label='hateful')
        self.da['TA1'] = text_df.describe()
        
        pyplot.legend()
        pyplot.savefig(self.dfp/'text_len.png')
        return None

    # Train/Val/Test split functions
    def load_given_data(self):
        """
        Assumes file names inside data directory to be: 
            train.jsonl, dev_unseen.jsonl, test_unseen.jsonl
        returns pandad df of already provided splits for train and test images
        """
        # Set file paths
        train_path = self.dfp / 'train.jsonl'
        val_path = self.dfp / 'dev_unseen.jsonl'
        test_path = self.dfp / 'test_unseen.jsonl'

        # load into panda data frames
        data_tr = pd.read_json(train_path, lines=True)
        data_v = pd.read_json(val_path, lines=True)
        data_t = pd.read_json(test_path, lines=True)
        
        return data_tr, data_v, data_t 
    
    def smart_select(self):
        # TODO
        return None