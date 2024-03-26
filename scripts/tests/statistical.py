import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/../dev/')

from metrics import MetricsTester
from constants import protected_attributes, group_proxies
from inference_engine import InferenceEngine




def preprocess(data, target_label='checked'):
    
    # Define your features and target
    X = data.drop(target_label, axis=1)
    y = data[target_label]

    return X, y


class StatisticalEvaluation:
    
    
    def __init__(self):
        pass
    
    
    def plot_classification_by_feature(self, X_test, y_test, y_pred, feature_name, feature_map=None):
        """
        Plots true negatives, true positives, false negatives, and false positives grouped by a feature.

        Args:
            model: Trained GradientBoostingClassifier model.
            X_test: Pandas dataframe containing the test features.
            y_test: Pandas Series containing the true labels for the test set.
            feature_name: Name of the feature to group classifications by.
        """

        # Store counts for each classification type and feature value
        class_counts = {'TP': {0: 0, 1: 0}, 'TN': {0: 0, 1: 0}, 'FP': {0: 0, 1: 0}, 'FN': {0: 0, 1: 0}}
        for i, (y_true, y_pred, feature_value) in enumerate(zip(y_test, y_pred, X_test[feature_name])):
            if y_true == 1 and y_pred == 1:
                class_counts['TP'][feature_value] += 1
            elif y_true == 1 and y_pred == 0:
                class_counts['FN'][feature_value] += 1
            elif y_true == 0 and y_pred == 1:
                class_counts['FP'][feature_value] += 1
            else:
                class_counts['TN'][feature_value] += 1

        # Extract data for plotting
        unique_features = sorted(set(X_test[feature_name]))

        # Create the figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))


        features = []
        if feature_map != None:
            
            for feature_value in unique_features:
                features.append(feature_map[feature_value])
        else:
            features = unique_features
        
        
 # Plot true positives
        sns.barplot(x=features, y=[class_counts['TP'][v] for v in unique_features], ax=axes[0, 0])
        axes[0, 0].set_xlabel(feature_name)
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('True Positives')

        # Plot true negatives
        sns.barplot(x=features, y=[class_counts['TN'][v] for v in unique_features], ax=axes[0, 1])
        axes[0, 1].set_xlabel(feature_name)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('True Negatives')

        # Plot false positives
        sns.barplot(x=features, y=[class_counts['FP'][v] for v in unique_features], ax=axes[1, 0])
        axes[1, 0].set_xlabel(feature_name)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('False Positives')

        # Plot false negatives
        sns.barplot(x=features, y=[class_counts['FN'][v] for v in unique_features], ax=axes[1, 1])
        axes[1, 1].set_xlabel(feature_name)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('False Negatives')

        # Common x labels for all subplots
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    ds_train = pd.read_csv('./../data/train.csv')
    ds_test = pd.read_csv('./../data/test.csv')    

    X_train, y_train = preprocess(ds_train)
    X_test, y_test = preprocess(ds_test)
    
    # Instantiate the ModelClass with ONNX model
    engine = InferenceEngine(onnx_model_path="./../model/good_model.onnx")
    
    y_pred = engine.predict(X_test)

    stats_metrics = StatisticalEvaluation()
    metrics = MetricsTester(protected_variables=protected_attributes)
    X_test_fair = metrics.preprocess_fairness_testing(X_test)
    stats_metrics.plot_classification_by_feature(X_test_fair, y_test, y_pred, group_proxies[0], feature_map={0: 'Male', 1: 'Female'})
    stats_metrics.plot_classification_by_feature(X_test_fair, y_test, y_pred, group_proxies[1], feature_map={0: 'Young', 1: 'Old'})
    stats_metrics.plot_classification_by_feature(X_test_fair, y_test, y_pred, group_proxies[2], feature_map={0: 'Dutch', 1: 'Non-Dutch'})
    stats_metrics.plot_classification_by_feature(X_test_fair, y_test, y_pred, group_proxies[3], feature_map={0: 'No-Children', 1: 'Children'})
        