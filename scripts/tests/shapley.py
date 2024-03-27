import shap  # package used to calculate Shap values


class ShapleyTester:

    def __init__(self):
        pass

    def get_tree_explainer(self, model):
        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(model)
        return explainer

    def get_model_explainer(self, model, X):
        # use Kernel SHAP to explain test set predictions
        data_summary = shap.kmeans(X, 10)
        explainer = shap.KernelExplainer(model.predict_proba, data_summary)
        return explainer

    def get_shapley_values(self, explainer, datapoint):
        k_shap_values = explainer.shap_values(datapoint)
        return k_shap_values

    def plot_shapley_values(self, explainer, datapoint):
        values = explainer.shap_values(datapoint)
        shap.initjs()
        return shap.force_plot(explainer.expected_value[0], values[1], datapoint)

