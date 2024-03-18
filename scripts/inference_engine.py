from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# onnx imports
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn


from preprocessing import preprocess

N_FEATURES_BASELINE_MODEL = 315


class InferenceEngine:
    def __init__(self, model_type='GB', onnx_model_path=None, pipeline=None):
        if model_type == 'GB':
            if pipeline == None:
                self.model = Pipeline(steps=[('classification', GradientBoostingClassifier(n_estimators=300, min_samples_split=800, min_samples_leaf=120, max_depth=5, learning_rate=0.145))])
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
                                         initial_types=[('X', FloatTensorType((None, N_FEATURES_BASELINE_MODEL)))],
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

    X_train, y_train = preprocess(ds_train)
    X_test, y_test = preprocess(ds_test)
    
    # Example usage
    # Instantiate the ModelClass with GradientBoostingClassifier
    engine = InferenceEngine(model_type='GB')
    engine.fit(X_train, y_train)
    y_pred_gb = engine.predict(X_test)

    engine.save_onnx_model(file_path="./../model/baseline_model.onnx")

    # Instantiate the ModelClass with ONNX model
    engine = InferenceEngine(model_type='ONNX', onnx_model_path="./../model/baseline_model.onnx")
    
    y_pred_baseline = engine.predict(X_test)
    
    from evaluation import EvaluationEngine
    evaluator = EvaluationEngine()
    
    evaluator.evaluate_generic_metrics(y_true=y_test, y_pred=y_pred_baseline)
