from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


# onnx imports
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

from testers.metrics import MetricsTester
from constants import protected_attributes, group_proxies

from preprocessing import preprocess

N_FEATURES_BASELINE_MODEL = 315


class InferenceEngine:
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


if __name__ == "__main__":


    ds_train = pd.read_csv('./../data/train.csv')
    ds_test = pd.read_csv('./../data/test.csv')    

    X_train, y_train = preprocess(ds_train)
    X_test, y_test = preprocess(ds_test)
    
    # Instantiate the ModelClass with ONNX model
    engine = InferenceEngine(onnx_model_path="./../model/good_model.onnx")
    
    y_pred = engine.predict(X_test)
    
    metrics = MetricsTester(protected_variables=group_proxies)
    X_test_fair = metrics.preprocess_fairness_testing(X_test)
    metrics.get_metrics_summary(X_test_fair, y_test, y_pred)
    
