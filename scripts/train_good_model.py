import pandas as pd
import numpy as np


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from sklearn.ensemble import GradientBoostingClassifier
from fairlearn.preprocessing import CorrelationRemover
from sklearn.metrics import make_scorer


from inference_engine import InferenceEngine
from constants import protected_attributes, group_proxies
from preprocessing import preprocess

from testers.metrics import MetricsTester
from model_training_environment import ModelWrapper, drop_feature

# Settings
np.random.seed(0)
import warnings
warnings.filterwarnings("ignore")

def custom_scoring(y_true, y_pred, fpr_threshold=0.005, fnr_treshold=0.5):
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
    false_negatives = (y_pred == 0) & (y_true == 1)
    false_negative_rate = false_negatives.sum() / len(y_true)
    score = recall * (1-fpr_threshold)
    if false_positive_rate > fpr_threshold or false_negative_rate < fnr_treshold:
        score = score **2

    return score


def run_hyperparameter_optimization():
    

    params = {
         'learning_rate':[0.15,0.1,0.05], 
         'n_estimators':[100,250,300, 350],
         "min_samples_split": [100, 250, 500, 800], 
         "min_samples_leaf":[25, 125, 200],
         "max_depth":[3, 5],
    }

    ds_train = pd.read_csv('./../data/train.csv')
    ds_test = pd.read_csv('./../data/test.csv')    

    X_train, y_train = preprocess(ds_train)
    X_test, y_test = preprocess(ds_test)
    
    for protected_variable in protected_attributes:
        X_train.loc[:, protected_variable] = X_train[protected_variable].apply(drop_feature)
        
    instance_weights = pd.read_csv('./../data/instance_weights_age_only2.csv')['instance_weights']


    score = make_scorer(custom_scoring, greater_is_better=True)
    tuning = RandomizedSearchCV(GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=42), 
                params, scoring=score, n_jobs=1, cv=5)
        
    tuning.fit(X_train,y_train)
    print(tuning.cv_results_)
    print('best:', tuning.best_params_)
    print('score:', tuning.best_score_)

    model = ModelWrapper(params=tuning.best_params_, protected_attributes=protected_attributes)
    model.fit(X_train, y_train, sample_weights=instance_weights)
    y_pred = model.predict(X_test)
    
    metrics = MetricsTester(protected_variables=group_proxies)
    X_test_fair = metrics.preprocess_fairness_testing(X_test)
    metrics.get_metrics_summary(X_test_fair, y_test, y_pred=y_pred)



    

def run(protected_variables, n_iterations=1):
        
    for model_id in range(n_iterations):
        # Train-test split
        
        df = pd.read_csv('./../data/synth_data_for_training.csv')
        X, y = preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)      
        X_test_fair = X_test.drop(protected_variables, axis=1)
        cr_alpha = CorrelationRemover(sensitive_feature_ids=protected_variables, alpha=0.75)
        
        X_cr_alpha = cr_alpha.fit_transform(X_train)

        # # Train model
        print(f'fitting model {model_id}...')
        model = GradientBoostingClassifier(n_estimators=350, min_samples_split=250, min_samples_leaf=25, max_depth=5, loss='log_loss', learning_rate=0.15, max_features='sqrt')
        
        model.fit(X_cr_alpha, y_train)
        y_pred = model.predict(X_test_fair)
        
        metrics = MetricsTester(protected_variables=protected_variables)
        metrics.get_metrics_summary(X_test, y_test, y_pred=y_pred)


if __name__ == "__main__":
    """
    
    1. do preprocessing
    2. do fairness contraints
    3. save model
    """
    
    run_hyperparameter_optimization()