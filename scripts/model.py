from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


N_FEATURES = 315

# onnx imports
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn

class InferenceEngine:
    def __init__(self, model_type='GB', onnx_model_path=None, pipeline=None):
        if model_type == 'GB':
            if pipeline == None:
                self.model = Pipeline(steps=[('classification', GradientBoostingClassifier())])
            else: 
                self.model = pipeline
        elif model_type == 'ONNX' and onnx_model_path:
            self.load_onnx_model(onnx_model_path)
        else:
            raise "No model loaded!"

    def fit(self, X_train, y_train):
        if isinstance(self.model, Pipeline):
            self.model.fit(X_train, y_train)
        else:
            raise ValueError("ONNX model does not support fit method")

    def predict(self, X_test):
        if isinstance(self.model, Pipeline):
            return self.model.predict(X_test)
        else:
            input_name = self.model.get_inputs()[0].name
            return self.model.run(None, {input_name: X_test.values.astype(np.float32)})[0]

    def save_onnx_model(self, file_path="./../model/baseline_model.onnx"):
        if isinstance(self.model, Pipeline):
            # Let's convert the model to ONNX
            onnx_model = convert_sklearn(self.model, 
                                         initial_types=[('X', FloatTensorType((None, N_FEATURES)))],
                                         target_opset=12)
            onnx.save(onnx_model, file_path)

        elif isinstance(self.model, rt.InferenceSession):
            # Save the ONNX model using onnx
            onnx.save_model(self.model, file_path)
        else:
            raise ValueError("Model type not supported for saving")

    def load_onnx_model(self, onnx_model_path):
        self.model = rt.InferenceSession(onnx_model_path)


if __name__ == "__main__":


    ds_train = pd.read_csv('./../data/train.csv')
    ds_test = pd.read_csv('./../data/test.csv')
    
    # Let's specify the features and the target
    y_train = ds_train["checked"]
    X_train = ds_train.drop(['checked'], axis=1)
    X_train = X_train.astype(np.float32)

    # Let's specify the features and the target
    y_test = ds_test["checked"]
    X_test = ds_test.drop(['checked'], axis=1)
    X_test = X_test.astype(np.float32)
    
    # Example usage
    # Instantiate the ModelClass with GradientBoostingClassifier
    engine = InferenceEngine(model_type='GB')
    engine.fit(X_train, y_train)
    y_pred_gb = engine.predict(X_test)

    engine.save_onnx_model(file_path="./../model/baseline_model_test.onnx")

    # Instantiate the ModelClass with ONNX model
    engine = InferenceEngine(model_type='ONNX', onnx_model_path="./../model/baseline_model_test.onnx")
    y_pred_onnx = engine.predict(X_test)
