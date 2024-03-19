import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def load_dataset(file_path):
    # Let's load the dataset
    data = pd.read_csv(file_path)

    return data

def get_dataset_split(X, y):
    # Let's split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test 

def preprocess():
    pass



def inprocess():
    pass
    
    
def postprocess():
    pass
    



if __name__ == "__main__":
    
    data = load_dataset('data/synth_data_for_training.csv')

    # Let's specify the features and the target
    y = data['checked']
    X = data.drop(['checked'], axis=1)
    X = X.astype(np.float32)

    data_positive_targets = data.loc[data['checked'] == 1]    
    X_train, X_test, y_train, y_test = get_dataset_split(X, y)