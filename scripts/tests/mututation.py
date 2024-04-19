import numpy as np
import pandas as pd

N_CLASSES = 2

def mutation(key):
    """
    This decorator is used to associate a key with a method that represents a mutation operation.

    Args:
        key (str): A unique key that identifies the mutation operation.

    Returns:
        callable: A decorator that can be applied to a method.
    """
  
    def do_assignment(to_func):
        to_func.key = key
        return to_func
    return do_assignment



class MutationTester:
    """
    This class is used to test the robustness of a machine learning model by applying various mutation operations
    to the training data and evaluating the impact on the model's performance.

    Attributes:
        model (object): The machine learning model to be tested.
        y_pred_baseline (np.ndarray): The baseline predictions of the model on the test data.
        df_train (pd.DataFrame): The training data.
        df_test (pd.DataFrame): The test data.
    """
  
    def __init__(self, model, y_pred_baseline, df_train, df_test):
        self.model = model
        self.y_pred_baseline = y_pred_baseline
        self.df_train = df_train
        self.df_test = df_test
    
    
    def test_mutants(self, params, n_mutants=5):
        """
        This method tests the mutation score of the model for various mutation operations.

        Args:
            params (dict): A dictionary containing parameters for each mutation operation.
            n_mutants (int, optional): The number of times to apply each mutation operation. Defaults to 5.

        Returns:
            dict: A dictionary containing the mutation scores for each mutation operation.
        """
    
        mutants = [getattr(self, field) for field in dir(self) if hasattr(getattr(self, field), "key")]
        mutation_scores = {}
        
        for mutant in mutants:
            
                n_mutations_killed = 0
                print(f"Testing {mutant.key}")
                for _ in range(n_mutants):
                    data = mutant(params[mutant.key])
                    X_train_mutant = data.iloc[:, :-1]
                    y_train_mutant = data.iloc[:, -1]
                    
    
                    mutant_model = self.model
                    mutant_model.fit(X_train_mutant, y_train_mutant)
                    
                    X_test = self.df_test.iloc[:, :-1]
                    y_test = self.df_test.iloc[:, -1]
                    y_pred_mutant = mutant_model.predict(X_test)

                    """
                        REFERENCE: https://sci-hub.se/10.1109/icst46399.2020.00018
                        The mutation score metric in DeepMutation is defined only for classification systems. 
                        More specifically, in case of a kclassification problem with a set of classes C = {c1, ..., ck}, 
                        a test input t ∈ T kills the pair c, m, with class c ∈ C and mutant m ∈ M, 
                        if t is correctly classified as c by the original model and if t is misclassified by the mutant model m. 
                        Based on this, the mutation score is calculated as the ratio of killed classes per mutant m over the product of the sizes of M and C
                    """
                    killed = (y_pred_mutant != y_test) & (self.y_pred_baseline == y_test)
                    
                    n_mutations_killed += sum(killed)
                mutation_scores[mutant.key] = n_mutations_killed / (n_mutants * N_CLASSES)
        
        return mutation_scores
                
    """
    The given training data is shuffled in a random order. Specifically, a random data point is swapped with another, 
    this is done equal to the iterations parameter number of times 
    """
    @mutation('data_shuffler')
    def data_shuffler(self, iterations=1000):
        """
        This method shuffles the training data by randomly swapping pairs of rows a specified number of times.

        Args:
            iterations (int, optional): The number of times to swap pairs of rows. Defaults to 1000.

        Returns:
            pd.DataFrame: A new DataFrame containing the shuffled training data.
        """
    
        shuffled_data = self.df_train.copy()
        num_rows = len(self.df_train)
        
        for i in range(iterations):
            # Choose two random indices for swapping
            idx1, idx2 = np.random.randint(0, num_rows, 2)
    
            # Swap rows at random two indices
            shuffled_data.iloc[idx1], shuffled_data.iloc[idx2] = (
                shuffled_data.iloc[idx2].copy(),
                shuffled_data.iloc[idx1].copy(),
            )
        return shuffled_data


    @mutation('data_remover')
    def data_remover(self, percent=0.1):
        """
        This method removes a fixed percentage of the training data at random.

        Args:
            percent (float, optional): The percentage of training data to remove. Defaults to 0.1.

        Returns:
            pd.DataFrame: A new DataFrame containing the reduced training data.
        """
        
        # Calculate the number of rows to remove based on the percentage
        num_rows_to_remove = int(len(self.df_train) * percent)
        smaller_df = self.df_train.copy()
        return smaller_df.drop(smaller_df.sample(n=num_rows_to_remove).index)
    

    @mutation('data_repetition')
    def data_repetition(self, iterations=1000):
        """
        This method randomly replicates the values of one row of the training data into another row a specified number of times.

        Args:
            iterations (int, optional): The number of times to repeat the data replication process. Defaults to 1000.

        Returns:
            pd.DataFrame: A new DataFrame containing the training data with repeated rows.
        """
        repeated_data = self.df_train.copy()
        num_rows = len(self.df_train)
        
        for i in range(iterations):
            # Choose two random indices. The value of the first will be duplicated onto the 2nd
            idx1, idx2 = np.random.randint(0, num_rows, 2)
            repeated_data.iloc[idx2] = repeated_data.iloc[idx1].values
        return repeated_data
    
    @mutation('label_error')
    def label_error(self, num_rows=1000):
        """
        This method randomly flips the labels (target values) of a specified number of rows in the training data.

        Args:
            num_rows (int, optional): The number of rows to alter the labels for. Defaults to 1000.

        Raises:
            AssertionError: If the number of rows to change labels for is greater than or equal to the total number of rows in the training data.

        Returns:
            pd.DataFrame: A new DataFrame containing the training data with flipped labels for some rows.
        """
        assert num_rows < len(self.df_train), "Please chose to alter the labels of less training data"
        changed_data = self.df_train.copy()
        # Get the indices of rows that will have their labels changed
        rows_to_change = np.random.choice(self.df_train.index, num_rows, replace=False)
        
        # Change the values of the last column ("checked") for the randomly selected rows (0 becomes 1 and vice versa)
        changed_data.loc[rows_to_change, self.df_train.columns[-1]] = 1 - changed_data.loc[rows_to_change, self.df_train.columns[-1]]
        
        return changed_data
    
    @mutation('feature_remover')
    def feature_remover(self, n_features=31):
        """
        This method randomly removes a specified number of features from the training data.

        Args:
        n_features (int, optional): The number of features to remove. Defaults to 31.

        Returns:
        pd.DataFrame: A new DataFrame containing the training data with removed features (set to zero).
        """
        new_dataset = self.df_train.copy()
        feature_subset = new_dataset.iloc[:, :-1].columns
        
        drop_features = np.random.choice(feature_subset, n_features)
        # We set all values to zero as a proxy for dropping the features. This allows for easier comparison.
        for col in drop_features: 
            new_dataset[col].values[:] = 0
        return new_dataset
    
    
if __name__ == "__main__":
    import onnxruntime as rt

    params = {
        'data_shuffler': 1000,
        'data_remover': 0.25,
        'data_repetition': 1000, 
        'label_error': 1000,
        'feature_remover': 31,
    }


    df_train = pd.read_csv("./../../data/datasets/train.csv")
    df_test = pd.read_csv("./../../data/datasets/test.csv")
    session = rt.InferenceSession("./../model/good_model.onnx")

    y_pred = session.run(None, {'X': df_test.iloc[:, :-1].values.astype(np.float32)})[0]

    mutator = MutationTester(y_pred_baseline=y_pred, df_train=df_train, df_test=df_test)
    mutation_score = mutator.test_mutants(params)
    print(f"Mutation score: {mutation_score}")
