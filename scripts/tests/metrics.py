import numpy as np
import pandas as pd

import onnxruntime as rt

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, false_positive_rate, false_negative_rate, selection_rate, equalized_odds_difference
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, fbeta_score, make_scorer
from utils.constants import group_proxies    


def classification_metrics(y_true, y_pred):
    print('y_true',y_true.shape)
    print('preed', y_pred.shape)
    print(confusion_matrix(y_true, y_pred))
    (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred).ravel()
    return (tn, fp, fn, tp)
    
    
def _precision_score(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=np.nan)
    
class MetricsTester():
    
    def __init__(self, protected_variables, group_metrics=None):
        
        self.functional_metrics = {
            "TN-FP-FN-TP": classification_metrics,
            'acc': accuracy_score,
            'prec': _precision_score, 
            'rec': recall_score,
            'f1': f1_score,
        }
        
        self.group_metrics = group_metrics if group_metrics is not None else {
            'fnr' : false_negative_rate,
            'fpr' : false_positive_rate,
            'sel' : selection_rate,
            'count' : count
        }
        
        self.overall_metrics = dict(self.functional_metrics, **self.group_metrics)
        self.protected_variables = protected_variables
    
    
    def get_metrics(self, y_true, y_pred, sensitive_features, is_group=True):
            
        print(y_true.shape)
        print(y_pred.shape)
        print(sensitive_features.shape)
        print(is_group)
        print('sens', sensitive_features.shape)
        mf = MetricFrame(
            metrics=self.group_metrics if is_group else self.functional_metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        return mf
    
    def get_metrics_by_group(self, y_true, y_pred, sensitive_features):

        mf = self.get_metrics(y_true, y_pred, sensitive_features, is_group=True)    
        return mf.by_group
    
    def get_metrics_overall(self, y_true, y_pred, sensitive_features):
        mf = self.get_metrics(y_true, y_pred, sensitive_features=sensitive_features, is_group=False)    
        return mf.overall
    
    def get_metrics_summary(self, X, y_true, y_pred):

        metrics_overall = self.get_metrics_overall(y_true=y_true, y_pred=y_pred, sensitive_features=X[self.protected_variables[0]])
        print(f"""Overal metrics: \n{metrics_overall} """)
        
        for group in self.protected_variables:
            metrics_group = self.get_metrics_by_group(y_true, y_pred, X[group])
            print(f"""Metrics for group: \n{metrics_group}""")
    
    
    def preprocess_fairness_testing(self, X):
        
        # Define a function to conditionally change age values
        def change_age_category(x):
            if x <= 27:
                return 0
            else:
                return 1

        # Define a function to conditionally change age values
        def change_language_category(x):
                if x == 57:
                    return 0
                else:
                    return 1
                
        # Define a function to conditionally change age values
        def modify_kids(x):
                if x == 0:
                    return 0
                else:
                    return 1

        # Define a function to conditionally change age values
        def modify_married_status(x):
                if x == 0:
                    return 0
                else:
                    return 1

        X_metrics = X.copy()
        X_metrics.loc[:, 'persoon_leeftijd_bij_onderzoek'] = X_metrics['persoon_leeftijd_bij_onderzoek'].apply(change_age_category)
        X_metrics.loc[:, 'persoonlijke_eigenschappen_spreektaal'] = X_metrics['persoonlijke_eigenschappen_spreektaal'].apply(change_language_category)
        X_metrics.loc[:, 'relatie_kind_huidige_aantal'] = X_metrics['relatie_kind_huidige_aantal'].apply(modify_kids)
        X_metrics.loc[:, 'relatie_partner_aantal_partner___partner__gehuwd_'] = X_metrics['relatie_partner_aantal_partner___partner__gehuwd_'].apply(modify_married_status)

        return X_metrics

if __name__ == "__main__":

    
    model_under_test = "./../model/good_model.onnx"
    
    df_test = pd.read_csv('./../data/test.csv')
    
    X_test = df_test.drop(['checked'], axis=1)
    y = df_test['checked']
    
    session = rt.InferenceSession(model_under_test)

    y_pred = session.run(None, {'X': df_test.iloc[:, :-1].values.astype(np.float32)})[0]    
    tester = MetricsTester(protected_variables=group_proxies)
    # X_test = tester.preprocess_fairness_testing(X_test)
    tester.get_metrics_summary(X=X_test, y_true=y,y_pred=y_pred)
        