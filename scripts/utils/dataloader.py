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

PATH_SYNTHETIC = ".\\..\\data\\datasets\\synth_data_for_training.csv"
PATH_TRAIN = ".\\..\\data\\datasets\\train.csv"
PATH_TEST = ".\\..\\data\\datasets\\test.csv"
PATH_TRAIN_GOOD = ".\\..\\data\\datasets\\train_good.csv"
PATH_TRAIN_BAD = ".\\..\\data\\datasets\\train_badly.csv"


DROPPED_FEATURE_CODE = 999

def drop_feature(x):
    """Replaces a feature value with a special code indicating the feature was dropped.

    Args:
        x: The value of the feature to be replaced.

    Returns:
        The special code indicating the feature was dropped (defined by DROPPED_FEATURE_CODE).
    """
    
    return DROPPED_FEATURE_CODE

class DataLoader():
    """Loads and splits datasets for machine learning tasks.

    This class provides methods to load various splits of a dataset (train, test, full, good, bad)
    and perform preprocessing steps like oversampling and dropping protected features.

    Attributes:
        dataset (pandas.DataFrame): The full dataset loaded from a CSV file (specified by PATH_SYNTHETIC).
        dataset_train (pandas.DataFrame): The training split of the dataset loaded from a CSV file (specified by PATH_TRAIN).
        dataset_test (pandas.DataFrame): The test split of the dataset loaded from a CSV file (specified by PATH_TEST).
        dataset_good (pandas.DataFrame): A subset of the training data containing "good" examples (specified by PATH_TRAIN_GOOD).
        dataset_bad (pandas.DataFrame): A subset of the training data containing "bad" examples (specified by PATH_TRAIN_BAD).
    """    
    def __init__(self):
        """
        Reads datasets from CSV files based on predefined paths.

        This constructor initializes the class attributes with pre-defined file paths for the full dataset,
        training, test, good, and bad splits. It's recommended to customize these paths for your specific project.

        Raises:
            ValueError: If a specified data file path is not found.
        """
        self.dataset = pd.read_csv(PATH_SYNTHETIC)
        self.dataset_train = pd.read_csv(PATH_TRAIN)
        self.dataset_test = pd.read_csv(PATH_TEST)
        self.dataset_good = pd.read_csv(PATH_TRAIN_GOOD)
        self.dataset_bad = pd.read_csv(PATH_TRAIN_BAD)
        
    def load_split(self, type='train'):
        """Loads a specific split of the dataset.

        This method takes a string argument indicating the desired data split ('train', 'test', 'full', 'good', or 'bad')
        and returns the corresponding features (X) and labels (y) as separate pandas DataFrames.

        Args:
            type (str, optional): The type of data split to load. Defaults to 'train'.

        Returns:
            tuple: A tuple containing two pandas DataFrames, the first containing features (X) and the second containing labels (y).

        Raises:
            Exception: If an invalid data split type is provided.
        """
        
        if type == 'train':
            data = self.dataset_train.copy()
        elif type == 'test':
            data = self.dataset_test.copy()
        elif type == 'full':
            data = self.dataset.copy()
        elif type == 'good':
            data = self.dataset_good.copy()
        elif type == 'bad':
            data = self.dataset_bad.copy()
        else:
            raise Exception
            
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        
        return X, y
    
    def load_and_split(self):
        """Loads the full dataset and splits it into features and labels using train_test_split.

        This method utilizes scikit-learn's train_test_split function to split the full dataset into features (X) and labels (y)
        with a 20% test set size and a random state of 2021 for reproducibility.

        Returns:
            tuple: A tuple containing two pandas DataFrames, the first containing training features (X_train) and the second containing training labels (y_train).
        """
        X = self.dataset.iloc[:,:-1]
        y = self.dataset.iloc[:,-1]
        return train_test_split(X, y, test_size=0.20, random_state=2021)

    def oversample(self, X_train, y_train):
            """Oversamples the training data to address class imbalance.

            This method utilizes scikit-learn's RandomOverSampler to oversample the minority class in the training data (X_train, y_train).
            Oversampling helps balance the class distribution and can improve model performance in classification tasks.

            Args:
                X_train (pandas.DataFrame): The training features.
                y_train (pandas.Series): The training labels.

            Returns:
                tuple: A tuple containing two pandas DataFrames, the first containing oversampled training features (X_over) 
                    and the second containing the corresponding oversampled training labels (y_over).
            """
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
        """Drops protected features from the dataset.

        This method removes specified features (protected_features) from the dataset (X) to comply with fairness regulations
        or prevent data leakage. It replaces the values of the dropped features with a special code (defined by DROPPED_FEATURE_CODE).

        Args:
            X (pandas.DataFrame): The dataset containing features.
            protected_features (list): A list of feature names to be dropped.

        Returns:
            pandas.DataFrame: A new DataFrame with protected features removed.
        """


        df = X.copy()
        for protected_variable in protected_features:
            df.loc[:, protected_variable] = df[protected_variable].apply(drop_feature)
            
        return df

    def save_split(self, X, y, path):
        """Saves a data split to a CSV file.

        This method combines the features (X) and labels (y) into a single DataFrame and saves it to a CSV file specified by the path argument.

        Args:
            X (pandas.DataFrame): The features for the data split.
            y (pandas.Series): The labels for the data split.
            path (str): The path to the directory where the CSV file will be saved.
        """

        data = pd.concat([X, y], axis=1)
        data.to_csv(path, index=False)
        print(f"Saved data to directory: {path}")