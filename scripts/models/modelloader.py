
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
        
    def load_model(type, params):
        if type == 'good':
            return GoodModel(params)
        elif type == 'bad':
            return BadModel(params)
        elif type == 'onnx':
            return OnnxModel(**params)    
            
    
class OnnxModel:
    def __init__(self, onnx_model_path=None):
        assert onnx_model_path is not None, 'ERROR: no model loaded!'
        if onnx_model_path:
            self.session = rt.InferenceSession(onnx_model_path,  providers=["CPUExecutionProvider"])
        else:
            raise "No model loaded!"

    def predict(self, dataset):
        """Predict the output using the wrapped regression model.
        :param dataset: The dataset to predict on.
        :type dataset: DatasetWrapper
        """
        y_pred = self.session.run(None, {'X': dataset.values.astype(np.float32)})[0]
        return np.array(y_pred)
    
    def predict_proba(self, X_test):
        y_pred_prob = self.session.run(None, {'X': X_test.values.astype(np.float32)})[1]
        probabilities = np.array([[item[0], item[1]] for item in y_pred_prob])
        return probabilities
    
    def load_onnx_model(self, onnx_model_path):
        self.model = rt.InferenceSession(onnx_model_path)
        
    def fit(self, X_train, y_train, sample_weights=None):
        print("ERROR: fit not implemented for onnx models!")        




class GoodModel:
    def __init__(self, params):
        self.model = GradientBoostingClassifier(**params) 
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
    
    