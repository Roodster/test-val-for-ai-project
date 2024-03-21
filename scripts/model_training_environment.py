from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


# onnx imports
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn


from preprocessing import preprocess
from testers.metrics import MetricsTester
from constants import protected_attributes, group_proxies
from inference_engine import InferenceEngine

N_FEATURES_BASELINE_MODEL = 315

DROPPED_FEATURE_CODE = 999

def drop_feature(x):
    return DROPPED_FEATURE_CODE




class ModelWrapper:
    def __init__(self, params, protected_attributes):
        self.model = GradientBoostingClassifier(**params) 
        self.protected_attributes = protected_attributes

    def fit(self, X_train, y_train, sample_weights=None):
        
        df_train = X_train.copy()
        
        for protected_variable in self.protected_attributes:
            df_train.loc[:, protected_variable] = df_train[protected_variable].apply(drop_feature)

        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        

    def predict(self, X_test):
        return self.model.predict(X_test)
   

    def save_onnx_model(self, file_path="./../model/baseline_model.onnx"):
        # Let's convert the model to ONNX
        onnx_model = convert_sklearn(self.model, 
                                        initial_types=[('X', FloatTensorType((None, N_FEATURES_BASELINE_MODEL)))],
                                        target_opset=12)
        onnx.save(onnx_model, file_path)




if __name__ == "__main__":


    ds_train = pd.read_csv('./../data/train.csv')
    ds_test = pd.read_csv('./../data/test.csv')    

    X_train, y_train = preprocess(ds_train)
    X_test, y_test = preprocess(ds_test)
    
    # Example usage
    # Instantiate the ModelClass with GradientBoostingClassifier

    params = {
        'n_estimators': 100, 
        'min_samples_split': 100, 
        'min_samples_leaf': 125, 
        'max_depth': 3, 
        'learning_rate': 0.05}
    
    instance_weights = pd.read_csv('./../data/instance_weights_age_only2.csv')['instance_weights']
    
    model = ModelWrapper(params=params, protected_attributes=protected_attributes)
    model.fit(X_train, y_train, sample_weights=instance_weights)
    y_pred_gb = model.predict(X_test)

    model.save_onnx_model(file_path="./../model/good_model_deprecated_2103.onnx")

    # Instantiate the ModelClass with ONNX model
    engine = InferenceEngine(model_type='ONNX', onnx_model_path="./../model/good_model_deprecated_2103.onnx")
    
    y_pred_new_model = engine.predict(X_test)

    metrics = MetricsTester(protected_variables=group_proxies)
    X_test_fair = metrics.preprocess_fairness_testing(X_test)
    metrics.get_metrics_summary(X_test_fair, y_test, y_pred=y_pred_new_model)