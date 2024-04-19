import shap  # package used to calculate Shap values
from interpret.ext.blackbox import TabularExplainer
import pandas as pd

class ShapleyTester:
    """
    This class provides functions for calculating and interpreting SHAP (SHapley Additive exPlanations) values
    to explain the predictions of a machine learning model.
    """
    def __init__(self):
        pass

    def get_tree_explainer(self, model):
        """
        This method creates a TabularExplainer object for tree-based models to calculate SHAP values.

        Args:
            model (object): The machine learning model to be explained.

        Returns:
            shap.TreeExplainer: A TabularExplainer object for tree-based models.
        """
        # Create object that can calculate shap values
        explainer = shap.TreeExplainer(model)
        return explainer

    def get_model_explainer(self, model, X):

        """
        This method creates a KernelExplainer object to explain model predictions using Kernel SHAP.

        Args:
            model (object): The machine learning model to be explained.
            X (pd.DataFrame): The feature data used to build the Kernel SHAP explainer.

        Returns:
            shap.KernelExplainer: A KernelExplainer object for explaining model predictions.
        """
        # use Kernel SHAP to explain test set predictions
        data_summary = shap.kmeans(X, 10)
        explainer = shap.KernelExplainer(model.predict_proba, data_summary)
        return explainer

    def get_shapley_values(self, explainer, datapoint):
        """
        This method calculates SHAP values for a single data point using the provided explainer object.

        Args:
            explainer (shap.Explainer): The explainer object used to calculate SHAP values.
            datapoint (pd.Series or np.ndarray): A single data point to explain.

        Returns:
            np.ndarray: The SHAP value vector for the data point.
        """
        
        k_shap_values = explainer.shap_values(datapoint)
        return k_shap_values

    def plot_shapley_values(self, explainer, datapoint):
        """
        This method creates a SHAP force plot to visualize the contribution of each feature to the model's prediction
        for a single data point.

        Args:
            explainer (shap.Explainer): The explainer object used to calculate SHAP values.
            datapoint (pd.Series or np.ndarray): A single data point to visualize.

        Returns:
            [type]: The SHAP force plot for the data point.
        """
        values = explainer.shap_values(datapoint)
        shap.initjs()
        return shap.force_plot(explainer.expected_value[0], values[1], datapoint)

    def run_shapley(self, model, train_data, test_data=None):
        """
        This method calculates SHAP values for a model using TabularExplainer and provides various outputs
        for global and local feature importances.

        Args:
            model (object): The machine learning model to be explained.
            train_data (pd.DataFrame): The training data used to fit the explainer.
            test_data (pd.DataFrame, optional): The test data to be explained (defaults to None, in which case
                                                train_data is used).

        Returns:
            tuple: A tuple containing the explainer object and the global explanation object.
        """
        
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
    
    def run_explain_instance(explainer, model, data, instance_id):
        """Runs explain_local method of explainer on a specified instance from the data.

        Args:
            explainer: An explainer object implementing explain_local.
            model: The machine learning model to be explained.
            data: A Pandas DataFrame containing the data to explain.
            instance_id: The index of the instance to explain in the data DataFrame.

        Returns:
            None. This function prints the local importance values and names for the explained instance.

        Raises:
            ValueError: If explainer does not support local explanations.
        """

        instance = data.iloc[instance_id,:]
        instance_df = pd.DataFrame(instance, data.columns).T

        local_explanation = explainer.explain_local(instance_df)
            
        # Get the prediction for the first member of the test set and explain why model made that prediction
        prediction_value = model.predict(data)[instance_id]

        sorted_local_importance_values = local_explanation.get_ranked_local_values()[prediction_value]
        sorted_local_importance_names = local_explanation.get_ranked_local_names()[prediction_value]

        print('local importance values: {}'.format(sorted_local_importance_values))
        print('local importance names: {}'.format(sorted_local_importance_names))
