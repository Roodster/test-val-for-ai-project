"""
CODE BASED ON: https://github.com/deepfindr/xai-series/blob/master/utils.py
"""
import os
import pandas as pd
import numpy as np
# Print all columns
pd.set_option('display.max_columns', None)
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

PATH_SYNTHETIC = "D:\\tudelft\\test-val-for-ai-project\data\datasets\synth_data_for_training.csv"
PATH_TRAIN = "D:\\tudelft\\test-val-for-ai-project\data\datasets\\train.csv"
PATH_TEST = "D:\\tudelft\\test-val-for-ai-project\data\datasets\\test.csv"


class DataLoader():
    
    def __init__(self):
        """
        TODO: add custom path to datasets option
        """
        
        self.dataset = pd.read_csv(PATH_SYNTHETIC)
        self.dataset_train = pd.read_csv(PATH_TRAIN)
        self.dataset_test = pd.read_csv(PATH_TEST)
        
    def load_split(self, type='train'):
        """ Loads dataset split according to type

        Args:
            type (str, optional): _description_. Defaults to 'train'.

        Returns:
            _type_: _description_
        """
        
        if type == 'train':
            data = self.dataset_train.copy()
        elif type == 'test':
            data = self.dataset_test.copy()
        elif type == 'full':
            data = self.dataset.copy()
            
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        
        return X, y
    
    def load_rand_split(self):
        X = self.dataset.iloc[:,:-1]
        y = self.dataset.iloc[:,-1]
        return train_test_split(X, y, test_size=0.20, random_state=2021)

    def oversample(self, X_train, y_train):
            oversample = RandomOverSampler(sampling_strategy='minority')
            # Convert to numpy and oversample
            x_np = X_train.to_numpy()
            y_np = y_train.to_numpy()
            x_np, y_np = oversample.fit_resample(x_np, y_np)
            # Convert back to pandas
            x_over = pd.DataFrame(x_np, columns=X_train.columns)
            y_over = pd.Series(y_np, name=y_train.name)
            return x_over, y_over