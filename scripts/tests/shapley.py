import shap  # package used to calculate Shap values


class ShapleyTester:

    def __init__(self):
        pass

    def get_tree_explainer(model, datapoint):
        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(model)
        return explainer

    def get_model_explainer(model, X):
        # use Kernel SHAP to explain test set predictions
        explainer = shap.KernelExplainer(model.predict_proba, X)
        return explainer

    def get_shapley_values(explainer, datapoint):
        k_shap_values = explainer.shap_values(datapoint)
        return k_shap_values

    def plot_shapley_values(explainer, datapoint):
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], explainer.shapley_values[1], datapoint)

