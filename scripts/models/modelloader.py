
import pandas as pd
import numpy as np

import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV


from sklearn.ensemble import GradientBoostingClassifier

from utils.constants import protected_attributes, group_proxies

N_FEATURES_BASELINE_MODEL = 315

DROPPED_FEATURE_CODE = 999

def drop_feature(x):
    return DROPPED_FEATURE_CODE

def custom_scoring(y_true, y_pred, fpr_threshold=0.005):
    """
    Custom scoring function that combines recall with false positive control.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        fp_threshold: Maximum tolerable false positive rate.

    Returns:
        A score combining recall and false positive control.
    """
    recall = recall_score(y_true, y_pred)
    false_positives = (y_pred == 1) & (y_true == 0)
    false_positive_rate = false_positives.sum() / len(y_true)

    penalty = 0
    if false_positive_rate > fpr_threshold:
        penalty = (false_positive_rate - fpr_threshold)**2

    return recall - penalty

class ModelLoader:
    
    def __init__(self, type='good', **params):
        
        if type == 'good':
            self.model = GoodModel(**params)
        elif type == 'bad':
            self.model = BadModel(**params)
        elif type == 'onnx':
            self.model = OnnxModel(**params)    
            
    def predict(self, X_test):
        self.model.predict(X_test)
    
    def fit(self, X_train, y_train, sample_weights=None):
        self.model.fit(X_train=X_train, y_train=y_train, sample_weights=sample_weights)
        
    def fit_hyperparameters(self, X_train, y_train, params, sample_weights=None):
        self.model.fit_hyperparameters(X_train, y_train, params, sample_weights)
        
    def save_onnx_model(self, file_path=""):
        assert file_path != "", "No file path"
        self.model.save_onnx_model(file_path)
    
class OnnxModel:
    def __init__(self, onnx_model_path=None):
        if onnx_model_path:
            self.load_onnx_model(onnx_model_path)
        else:
            raise "No model loaded!"


    def predict(self, X_test):
        input_name = self.model.get_inputs()[0].name
        return self.model.run(None, {input_name: X_test.values.astype(np.float32)})[0]

    def load_onnx_model(self, onnx_model_path):
        self.model = rt.InferenceSession(onnx_model_path)
        
    def fit(self, X_train, y_train, sample_weights=None):
        print("ERROR: fit not implemented for onnx models!")        


class GoodModel:
    def __init__(self, params, protected_attributes):
        self.model = GradientBoostingClassifier(**params) 
        self.protected_attributes = protected_attributes
        self.sample_weights = None

    def fit(self, X_train, y_train, sample_weights=None):    
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
    def fit_hyperparameters(self, X_train, y_train, params):
        
        ftwo_scorer = make_scorer(custom_scoring)
        tuning = RandomizedSearchCV(GradientBoostingClassifier(), 
                        params, scoring=ftwo_scorer, n_jobs=4, cv=5)
            
        tuning.fit(X_train,y_train)
        self.model = GradientBoostingClassifier(tuning.best_params_)
        
        self.fit(X_train=X_train, y_train=y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
   

    def save_onnx_model(self, file_path=""):
        assert file_path != "", "No file path"
        # Let's convert the model to ONNX
        onnx_model = convert_sklearn(self.model, 
                                        initial_types=[('X', FloatTensorType((None, N_FEATURES_BASELINE_MODEL)))],
                                        target_opset=12)
        onnx.save(onnx_model, file_path)
        print(f"Saved model to {file_path}")

class BadModel:
    pass
    
    