import aif360.sklearn as skm

class Wrapper:
    
   def get_metrics(self):
        metrics = {}
        for attr in dir(self):
            if isinstance(getattr(self.__class__, attr, None), property):
                metrics[attr] = getattr(self, attr)
        return metrics

class GenericMetricsWrapper(Wrapper):
    def __init__(self, y_true, y_pred, probas_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.probas_pred = probas_pred

    @property
    def num_samples(self):
        return skm.metrics.num_samples(self.y_true, self.y_pred)

    @property
    def num_pos_neg(self):
        return skm.metrics.num_pos_neg(self.y_true, self.y_pred)

    @property
    def specificity_score(self):
        return skm.metrics.specificity_score(self.y_true, self.y_pred)

    @property
    def sensitivity_score(self):
        return skm.metrics.sensitivity_score(self.y_true, self.y_pred)

    @property
    def base_rate(self):
        return skm.metrics.base_rate(self.y_true, self.y_pred)

    @property
    def selection_rate(self):
        return skm.metrics.selection_rate(self.y_true, self.y_pred)

    @property
    def smoothed_base_rate(self):
        return skm.metrics.smoothed_base_rate(self.y_true, self.y_pred)

    @property
    def smoothed_selection_rate(self):
        return skm.metrics.smoothed_selection_rate(self.y_true, self.y_pred)

    @property
    def generalized_fpr(self):
        return skm.metrics.generalized_fpr(self.y_true, self.probas_pred)

    @property
    def generalized_fnr(self):
        return skm.metrics.generalized_fnr(self.y_true, self.probas_pred)
class GroupMetricsWrapper:
    def __init__(self, y_true, y_pred, probas_pred, protected_attributes, dataset):
        self.y_true = y_true
        self.y_pred = y_pred
        self.probas_pred = probas_pred
        self.prot_attr = protected_attributes
        self.X = dataset

    @property
    def statistical_parity_difference(self):
        return skm.metrics.statistical_parity_difference(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def mean_difference(self):
        return skm.metrics.mean_difference(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def disparate_impact_ratio(self):
        return skm.metrics.disparate_impact_ratio(self.y_true)

    @property
    def equal_opportunity_difference(self):
        return skm.metrics.equal_opportunity_difference(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def average_odds_difference(self):
        return skm.metrics.average_odds_difference(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def average_odds_error(self):
        return skm.metrics.average_odds_error(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def class_imbalance(self):
        return skm.metrics.class_imbalance(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def kl_divergence(self):
        return skm.metrics.kl_divergence(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def conditional_demographic_disparity(self):
        return skm.metrics.conditional_demographic_disparity(self.y_true)

    @property
    def smoothed_edf(self):
        return skm.metrics.smoothed_edf(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def df_bias_amplification(self):
        return skm.metrics.df_bias_amplification(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def between_group_generalized_entropy_error(self):
        return skm.metrics.between_group_generalized_entropy_error(self.y_true, y_pred=self.y_pred, prot_attr=self.prot_attr)

    @property
    def mdss_bias_score(self):
        return skm.metrics.mdss_bias_score(self.y_true, self.probas_pred, self.X)
class IndividualMetricsWrapper(Wrapper):
    def __init__(self, y_true, y_pred, probas_pred, X, y, alpha=2, n_neighbors=5):
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
    #     return skm.metrics.generalized_entropy_index(self.b, self.alpha)

    @property
    def generalized_entropy_error(self):
        return skm.metrics.generalized_entropy_error(self.y_true, self.y_pred)

    # @property
    # def theil_index(self):
    #     return skm.metrics.theil_index(self.b)

    # @property
    # def coefficient_of_variation(self):
    #     return skm.metrics.coefficient_of_variation(self.b)

    # @property
    # def consistency_score(self):
    #     return skm.metrics.consistency_score(self.X, self.y, self.n_neighbors)



if __name__ == "__main__":
    
    # TODO: Fix example.
    pass
