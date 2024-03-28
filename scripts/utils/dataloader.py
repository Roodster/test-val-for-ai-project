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


DROPPED_FEATURE_CODE = 999

def drop_feature(x):
    return DROPPED_FEATURE_CODE

class DataLoader():
    
    def __init__(self):
        """
        TODO: add custom path to datasets option
        """
        
        self.dataset = pd.read_csv(PATH_SYNTHETIC)
        self.dataset_train = pd.read_csv(PATH_TRAIN)
        self.dataset_test = pd.read_csv(PATH_TEST)
        
    def load_split(self, type='train'):
        """ Loads dataset split according to type: train, test, full
        """
        
        if type == 'train':
            data = self.dataset_train.copy()
        elif type == 'test':
            data = self.dataset_test.copy()
        elif type == 'full':
            data = self.dataset.copy()
        else:
            raise Exception
            
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
    
    def drop_protected_features(self, X, protected_features):
        
        df = X.copy()
        for protected_variable in protected_features:
            df.loc[:, protected_variable] = df[protected_variable].apply(drop_feature)
            
        return df

    def save_split(X_train, y_train, path):
        data = pd.concat([X_train, y_train], axis=1)
        data.to_csv(data, path)
        print(f"Saved data to directory: {path}")