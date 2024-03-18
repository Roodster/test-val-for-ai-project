import unittest
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from scripts.inference_engine import InferenceEngine
from scripts.preprocessing import preprocess



class TestFunctional(unittest.TestCase):
    
    
    """
       @TEST: Test the accuracy of the trained model.
    """
    def test_accuracy_good_model(self):
        ds_test = pd.read_csv('./../data/test.csv')
        
        # Instantiate the ModelClass with ONNX model
        engine = InferenceEngine(model_type='ONNX', onnx_model_path="./../model/baseline_model.onnx")
        X_test, y_test = preprocess(ds_test)
        
        y_pred= engine.predict(X_test)

        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        self.assertTrue(accuracy >= 0.9)
    
    def test_accuracy_bad_model(self):
        pass
    
    def test_precision_good_model(self):    
        pass    

    def test_precision_bad_model(self):
        pass
    
    


if __name__ == "__main__":
    unittest.main()