import pandas as pd
import numpy as np

import aif360.sklearn.metrics as skm
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, classification_report


# Settings
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore")

class Wrapper:
    
   def get_metrics(self):
        metrics = {}
        for attr in dir(self):
            if isinstance(getattr(self.__class__, attr, None), property):
                metrics[attr] = getattr(self, attr)
        return metrics

class MetricWrapper(Wrapper):
    def __init__(self, metric):
        self.metric = metric
        
    @property
    def accuracy(self):
        return self.metric.accuracy()

    @property
    def average_abs_odds_difference(self):
        return self.metric.average_abs_odds_difference()

    @property
    def average_odds_difference(self):
        return self.metric.average_odds_difference()

    @property
    def average_predictive_value_difference(self):
        return self.metric.average_predictive_value_difference()

    @property
    def base_rate(self):
        return self.metric.base_rate()

    @property
    def between_all_groups_coefficient_of_variation(self):
        return self.metric.between_all_groups_coefficient_of_variation()

    @property
    def between_all_groups_generalized_entropy_index(self):
        return self.metric.between_all_groups_generalized_entropy_index()

    @property
    def between_all_groups_theil_index(self):
        return self.metric.between_all_groups_theil_index()

    @property
    def between_group_coefficient_of_variation(self):
        return self.metric.between_group_coefficient_of_variation()

    @property
    def between_group_generalized_entropy_index(self):
        return self.metric.between_group_generalized_entropy_index()

    @property
    def between_group_theil_index(self):
        return self.metric.between_group_theil_index()

    @property
    def binary_confusion_matrix(self):
        return self.metric.binary_confusion_matrix()

    @property
    def coefficient_of_variation(self):
        return self.metric.coefficient_of_variation()

    @property
    def consistency(self):
        return self.metric.consistency()

    # @property
    # def difference(self):
    #     return self.metric.difference()

    @property
    def differential_fairness_bias_amplification(self):
        return self.metric.differential_fairness_bias_amplification()

    @property
    def disparate_impact(self):
        return self.metric.disparate_impact()

    @property
    def equal_opportunity_difference(self):
        return self.metric.equal_opportunity_difference()

    @property
    def equalized_odds_difference(self):
        return self.metric.equalized_odds_difference()

    @property
    def error_rate(self):
        return self.metric.error_rate()

    @property
    def error_rate_difference(self):
        return self.metric.error_rate_difference()

    @property
    def error_rate_ratio(self):
        return self.metric.error_rate_ratio()

    @property
    def false_discovery_rate(self):
        return self.metric.false_discovery_rate()

    @property
    def false_discovery_rate_difference(self):
        return self.metric.false_discovery_rate_difference()

    @property
    def false_discovery_rate_ratio(self):
        return self.metric.false_discovery_rate_ratio()

    @property
    def false_negative_rate(self):
        return self.metric.false_negative_rate()

    @property
    def false_negative_rate_difference(self):
        return self.metric.false_negative_rate_difference()

    @property
    def false_negative_rate_ratio(self):
        return self.metric.false_negative_rate_ratio()

    @property
    def false_omission_rate(self):
        return self.metric.false_omission_rate()

    @property
    def false_omission_rate_difference(self):
        return self.metric.false_omission_rate_difference()

    @property
    def false_omission_rate_ratio(self):
        return self.metric.false_omission_rate_ratio()

    @property
    def false_positive_rate(self):
        return self.metric.false_positive_rate()

    @property
    def false_positive_rate_difference(self):
        return self.metric.false_positive_rate_difference()

    @property
    def false_positive_rate_ratio(self):
        return self.metric.false_positive_rate_ratio()

    @property
    def generalized_binary_confusion_matrix(self):
        return self.metric.generalized_binary_confusion_matrix()

    @property
    def generalized_entropy_index(self):
        return self.metric.generalized_entropy_index()

    @property
    def generalized_equalized_odds_difference(self):
        return self.metric.generalized_equalized_odds_difference()

    @property
    def generalized_false_negative_rate(self):
        return self.metric.generalized_false_negative_rate()

    @property
    def generalized_false_positive_rate(self):
        return self.metric.generalized_false_positive_rate()

    @property
    def generalized_true_negative_rate(self):
        return self.metric.generalized_true_negative_rate()

    @property
    def generalized_true_positive_rate(self):
        return self.metric.generalized_true_positive_rate()

    @property
    def mean_difference(self):
        return self.metric.mean_difference()

    @property
    def negative_predictive_value(self):
        return self.metric.negative_predictive_value()

    @property
    def num_false_negatives(self):
        return self.metric.num_false_negatives()

    @property
    def num_false_positives(self):
        return self.metric.num_false_positives()

    @property
    def num_generalized_false_negatives(self):
        return self.metric.num_generalized_false_negatives()

    @property
    def num_generalized_false_positives(self):
        return self.metric.num_generalized_false_positives()
    
    @property
    def recall(self):
        return self.metric.recall()
    
    @property
    def precision(self):
        return self.metric.precision()
    
    @property
    def num_generalized_true_negatives(self):
        return self.metric.num_generalized_true_negatives()

    @property
    def num_generalized_true_positives(self):
        return self.metric.num_generalized_true_positives()

    @property
    def num_instances(self):
        return self.metric.num_instances()

    @property
    def num_negatives(self):
        return self.metric.num_negatives()

    @property
    def num_positives(self):
        return self.metric.num_positives()

    @property
    def num_pred_negatives(self):
        return self.metric.num_pred_negatives()

    @property
    def num_pred_positives(self):
        return self.metric.num_pred_positives()

    @property
    def num_true_negatives(self):
        return self.metric.num_true_negatives()

    @property
    def num_true_positives(self):
        return self.metric.num_true_positives()

    @property
    def performance_measures(self):
        return self.metric.performance_measures()

    @property
    def positive_predictive_value(self):
        return self.metric.positive_predictive_value()

    @property
    def power(self):
        return self.metric.power()

    @property
    def precision(self):
        return self.metric.precision()

    # @property
    # def ratio(self):
    #     return self.metric.ratio()

    @property
    def recall(self):
        return self.metric.recall()

    # @property
    # def rich_subgroup(self):
    #     return self.metric.rich_subgroup()

    @property
    def selection_rate(self):
        return self.metric.selection_rate()

    @property
    def sensitivity(self):
        return self.metric.sensitivity()

    @property
    def smoothed_empirical_differential_fairness(self):
        return self.metric.smoothed_empirical_differential_fairness()

    @property
    def specificity(self):
        return self.metric.specificity()

    @property
    def statistical_parity_difference(self):
        return self.metric.statistical_parity_difference()

    @property
    def theil_index(self):
        return self.metric.theil_index()

    @property
    def true_negative_rate(self):
        return self.metric.true_negative_rate()

    @property
    def true_positive_rate(self):
        return self.metric.true_positive_rate()

    @property
    def true_positive_rate_difference(self):
        return self.metric.true_positive_rate_difference()
    
    def get_metrics(self):
        metrics = {}
        for attr in dir(self):
            if isinstance(getattr(self.__class__, attr, None), property):
                metrics[attr] = getattr(self, attr)
        return metrics



class GenericMetricsWrapper(Wrapper):
    def __init__(self, y_true, y_pred, probas_pred=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.probas_pred = probas_pred

    @property
    def num_samples(self):
        return skm.num_samples(self.y_true, self.y_pred)

    @property
    def num_pos_neg(self):
        return skm.num_pos_neg(self.y_true, self.y_pred)

    @property
    def specificity_score(self):
        return skm.specificity_score(self.y_true, self.y_pred)

    @property
    def sensitivity_score(self):
        return skm.sensitivity_score(self.y_true, self.y_pred)

    @property
    def base_rate(self):
        return skm.base_rate(self.y_true, self.y_pred)

    @property
    def selection_rate(self):
        return skm.selection_rate(self.y_true, self.y_pred)

    @property
    def smoothed_base_rate(self):
        return skm.smoothed_base_rate(self.y_true, self.y_pred)

    @property
    def smoothed_selection_rate(self):
        return skm.smoothed_selection_rate(self.y_true, self.y_pred)

    @property
    def classification_report(self):
        class_report = classification_report(self.y_true, self.y_pred)
        return class_report

    @property
    def confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        return conf_matrix


    # @property
    # def generalized_fpr(self):
    #     return skm.generalized_fpr(self.y_true, self.probas_pred)

    # @property
    # def generalized_fnr(self):
    #     return skm.generalized_fnr(self.y_true, self.probas_pred)
        
class GroupMetricsWrapper(Wrapper):
    def __init__(self, y_true, y_pred, protected_attributes, dataset, probas_pred=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.probas_pred = probas_pred
        self.prot_attr = protected_attributes
        self.X = dataset

    @property
    def statistical_parity_difference(self):
        return skm.statistical_parity_difference(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def mean_difference(self):
        return skm.mean_difference(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def disparate_impact_ratio(self):
        return skm.disparate_impact_ratio(self.y_true)

    @property
    def equal_opportunity_difference(self):
        return skm.equal_opportunity_difference(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def average_odds_difference(self):
        return skm.average_odds_difference(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def average_odds_error(self):
        return skm.average_odds_error(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def class_imbalance(self):
        return skm.class_imbalance(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def kl_divergence(self):
        return skm.kl_divergence(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def conditional_demographic_disparity(self):
        return skm.conditional_demographic_disparity(self.y_true)

    @property
    def smoothed_edf(self):
        return skm.smoothed_edf(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def df_bias_amplification(self):
        return skm.df_bias_amplification(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def between_group_generalized_entropy_error(self):
        return skm.between_group_generalized_entropy_error(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    # @property
    # def mdss_bias_score(self):
    #     return skm.mdss_bias_score(self.y_true, self.probas_pred, self.X)

class IndividualMetricsWrapper(Wrapper):
    def __init__(self, y_true, y_pred, X, y, probas_pred=None,  alpha=2, n_neighbors=5):
        self.y_true = y_true
        self.y_pred = y_pred
        self.probas_pred = probas_pred
        self.b = X.to_numpy()
        self.X = X
        self.y = y
        self.alpha = alpha
        self.n_neighbors = n_neighbors

    # @property
    # def generalized_entropy_index(self):
    #     return skm.generalized_entropy_index(self.b, self.alpha)

    @property
    def generalized_entropy_error(self):
        return skm.generalized_entropy_error(self.y_true, self.y_pred)

    # @property
    # def theil_index(self):
    #     return skm.theil_index(self.b)

    # @property
    # def coefficient_of_variation(self):
    #     return skm.coefficient_of_variation(self.b)

    # @property
    # def consistency_score(self):
    #     return skm.consistency_score(self.X, self.y, self.n_neighbors)

    def get_metrics(self):
        return super().get_metrics()

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
    from model import InferenceEngine
    # Instantiate the ModelClass with ONNX model
    engine = InferenceEngine(model_type='ONNX', onnx_model_path="./../model/good_model.onnx")
    y_pred = engine.predict(X_test)

    protected_attributes = X_test['persoon_geslacht_vrouw']
    generic_metrics = GenericMetricsWrapper(y_true=y_test, y_pred=y_pred)

    print('generic metrics: ')

    results = generic_metrics.get_metrics()

    for metric, value in results.items():
        print(f"""{metric:<30} {value}""")

    group_metrics = GroupMetricsWrapper(y_true=y_test, y_pred=y_pred, protected_attributes=protected_attributes, dataset=X_test)

    results = group_metrics.get_metrics()

    print('group  metrics:')
    # Print the results
    for metric, value in results.items():
        print(f"""{metric:<30} {value}""")

    indiv_metrics = IndividualMetricsWrapper(y_true=y_test, y_pred=y_pred, X=X_test, y=y_test, alpha=None, n_neighbors=None)

    results = indiv_metrics.get_metrics()
    print('individual metrics: ')

    # Print the results
    for metric, value in results.items():
        print(f"{metric:<30} {value}")
        
        
    """ TODO: fix metricwrapper example   
    # Metric for the original dataset
    privileged_groups = [{'persoon_geslacht_vrouw': 0.0}]
    unprivileged_groups = [{'persoon_geslacht_vrouw': 1.0}]


    metric = ClassificationMetric(dataset_test, dataset_pred, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    # Assuming metrics is already defined as an instance of MetricWrapper
    metrics = MetricWrapper(metric)

    results = metrics.get_metrics()
    # Print the results
    for metric, value in results.items():
        print(f"{metric}: {value}")"""
