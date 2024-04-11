import shap  # package used to calculate Shap values
from interpret.ext.blackbox import TabularExplainer
import pandas as pd

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

    def run_shapley(self, model, train_data, test_data=None):
        
        if test_data == None:
            test_data = train_data
        
        # 1. Using SHAP TabularExplainer
        explainer = TabularExplainer(model, 
                                    train_data, 
                                    features=train_data.columns, 
                                    classes=['fraud', 'no fraud'])

        # Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
        # x_train can be passed as well, but with more examples explanations will take longer although they may be more accurate
        global_explanation = explainer.explain_global(test_data)


        # Sorted SHAP values
        print('ranked global importance values: {}'.format(global_explanation.get_ranked_global_values()))
        # Corresponding feature names
        print('ranked global importance names: {}'.format(global_explanation.get_ranked_global_names()))
        # Feature ranks (based on original order of features)
        print('global importance rank: {}'.format(global_explanation.global_importance_rank))

        # Note: Do not run this cell if using PFIExplainer, it does not support per class explanations
        # Per class feature names
        print('ranked per class feature names: {}'.format(global_explanation.get_ranked_per_class_names()))
        # Per class feature importance values
        print('ranked per class feature values: {}'.format(global_explanation.get_ranked_per_class_values()))
        
        # Print out a dictionary that holds the sorted feature importance names and values
        print('global importance rank: {}'.format(global_explanation.get_feature_importance_dict()))
        
        # feature shap values for all features and all data points in the training data
        print('local importance values: {}'.format(global_explanation.local_importance_values))
        
        return explainer, global_explanation
    
    def run_explain_instance(explainer, data, instance_id):
        # Note: Do not run this cell if using PFIExplainer, it does not support local explanations
        # You can pass a specific data point or a group of data points to the explain_local function

        instance = data.iloc[instance_id,:]
        instance_df = pd.DataFrame(instance, data.columns).T

        local_explanation = explainer.explain_local(instance_df)
            
        # Get the prediction for the first member of the test set and explain why model made that prediction
        prediction_value = model.predict(X_test)[instance_id]

        sorted_local_importance_values = local_explanation.get_ranked_local_values()[prediction_value]
        sorted_local_importance_names = local_explanation.get_ranked_local_names()[prediction_value]

        print('local importance values: {}'.format(sorted_local_importance_values))
        print('local importance names: {}'.format(sorted_local_importance_names))
