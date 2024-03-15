import pandas as pd
import numpy as np

import onnxruntime as rt
import onnx


def load_model(model_path):
    # Let's load the model
    new_session = rt.InferenceSession(model_path)
    return new_session
    
def predict(model, data):
    # Let's predict the target
    y_pred =  model.run(None, {'X': data.values.astype(np.float32)})
    return y_pred


if __name__ == "__main__":
    
    model = load_model(model_path="model/gboost.onnx")
                       
    data = None
    
    y_preds = predict(model, data)