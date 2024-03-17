import pandas as pd
import numpy as np

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, \
                              false_positive_rate, \
                              selection_rate, equalized_odds_ratio
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
from fairlearn.reductions import (  # noqa
    DemographicParity,
    EqualizedOdds,
    ExponentiatedGradient,
)
from fairlearn.preprocessing import CorrelationRemover


from evaluation import EvaluationEngine
from inference_engine import InferenceEngine
from constants import protected_attributes
from preprocessing import preprocess

# Settings
np.random.seed(0)
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    """
    
    1. do preprocessing
    2. do fairness contraints
    3. save model
    """
    
    
        
    ds_train = pd.read_csv('./../data/train.csv')
    ds_test = pd.read_csv('./../data/test.csv')

    X_train, y_train = preprocess(ds_train)
    X_test, y_test = preprocess(ds_test)
    
    # cr_alpha = CorrelationRemover(sensitive_feature_ids=protected_attributes, alpha=0.5)
    
    # X_cr_alpha = cr_alpha.fit_transform(X_train)
    

    model = InferenceEngine(model_type='GB')
    # model.fit(X_cr_alpha, y_train)
    X_test_fair = X_test.drop(protected_attributes, axis=1)
    y_pred_fair = model.predict(X_test_fair)    


    baseline_model = InferenceEngine(model_type='ONNX', onnx_model_path='./../model/baseline_model.onnx')
    y_pred_baseline = baseline_model.predict(X_test)


    print("Evaluation of \'baseline\' model ")
    
    # Instantiate the ModelClass with ONNX model
    engine = InferenceEngine(model_type='ONNX', onnx_model_path="./../model/baseline_model.onnx")
    
    y_pred_baseline = engine.predict(X_test)
    
    evaluator = EvaluationEngine()
    
    evaluator.evaluate_generic(y_true=y_test, y_pred=y_pred_baseline)

    print("Evaluation of \'good\' model ")
    
    evaluator = EvaluationEngine()
     
    evaluator.evaluate_generic(y_test, y_pred_fair)
    # evaluator.evaluate_group(y_test, y_pred, X_test)
 