import numpy as np
import pandas as pd

from model_training_environment import ModelWrapper
from constants import protected_attributes
import onnxruntime as rt



def mutation(key):
    print(f"Initializing test: {key}...")
    def do_assignment(to_func):
        to_func.key = key
        return to_func
    return do_assignment



class MutationTester:
    def __init__(self, y_pred_baseline, df_train, df_test):
        self.y_pred_baseline = y_pred_baseline
        self.df_train = df_train
        self.df_test = df_test
    
    
    def test_mutants(self, params, n_mutants=5):
        
        mutants = [getattr(self, field) for field in dir(self) if hasattr(getattr(self, field), "key")]
        mutation_scores = {}
        
        for mutant in mutants:
            
                n_mutations_killed = 0
                print(f"Testing {mutant.key}")
                for _ in range(n_mutants):
                    data = mutant(params[mutant.key])
                    X_train_mutant = data.iloc[:, :-1]
                    y_train_mutant = data.iloc[:, -1]
                    
                    model_params ={
                        'n_estimators':350, 
                        'min_samples_split':800, 
                        "min_samples_leaf":200, 
                        "max_depth":5,
                        "learning_rate":0.15
                    }
                    mutant_model = ModelWrapper(model_params, protected_attributes=protected_attributes)
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
                    killed = (y_pred_mutant == y_test) & (self.y_pred_baseline != y_test)
                    
                    n_mutations_killed += sum(killed)
                mutation_scores[mutant.key] = n_mutations_killed / (n_mutants * 2)
        
        return mutation_scores
                
    """
    The given training data is shuffled in a random order. Specifically, a random data point is swapped with another, 
    this is done equal to the iterations parameter number of times 
    """
    @mutation('data_shuffler')
    def data_shuffler(self, iterations=1000):
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

    """
    A fixed percentage of the training data is removed (at random)
    """
    @mutation('data_remover')
    def data_remover(self, percent=0.1):
        # Calculate the number of rows to remove based on the percentage
        num_rows_to_remove = int(len(self.df_train) * percent)
        smaller_df = self.df_train.copy()
        return smaller_df.drop(smaller_df.sample(n=num_rows_to_remove).index)
    
    """
    Chosen randomly, the values of a row of the training data is replicated into another row. This is done
    iterations number of times
    """
    @mutation('data_repetition')
    def data_repetition(self, iterations=1000):
        repeated_data = self.df_train.copy()
        num_rows = len(self.df_train)
        
        for i in range(iterations):
            # Choose two random indices. The value of the first will be duplicated onto the 2nd
            idx1, idx2 = np.random.randint(0, num_rows, 2)
            repeated_data.iloc[idx2] = repeated_data.iloc[idx1].values
        return repeated_data
    
    @mutation('label_error')
    def label_error(self, num_rows=1000):
        assert num_rows < len(self.df_train), "Please chose to alter the labels of less training data"
        changed_data = self.df_train.copy()
        # Get the indices of rows that will have their labels changed
        rows_to_change = np.random.choice(self.df_train.index, num_rows, replace=False)
        
        # Change the values of the last column ("checked") for the randomly selected rows (0 becomes 1 and vice versa)
        changed_data.loc[rows_to_change, self.df_train.columns[-1]] = 1 - changed_data.loc[rows_to_change, self.df_train.columns[-1]]
        
        return changed_data
    
    @mutation('feature_remover')
    def feature_remover(self, n_features=31):
        new_dataset = self.df_train.copy()
        feature_subset = new_dataset.iloc[:, :-1].columns
        
        drop_features = np.random.choice(feature_subset, n_features)
        # We set all values to zero as a proxy for dropping the features. This allows for easier comparison.
        for col in drop_features: 
            new_dataset[col].values[:] = 0
        return new_dataset
    
    
if __name__ == "__main__":
    
    params = {
        'data_shuffler': 1000,
        'data_remover': 0.25,
        'data_repetition': 1000, 
        'label_error': 1000,
        'feature_remover': 31,
    }


    df_train = pd.read_csv("./../data/train.csv")
    df_test = pd.read_csv("./../data/test.csv")
    session = rt.InferenceSession("./../model/good_model.onnx")

    y_pred = session.run(None, {'X': df_test.iloc[:, :-1].values.astype(np.float32)})[0]

    mutator = MutationTester(y_pred_baseline=y_pred, df_train=df_train, df_test=df_test)
    mutation_score = mutator.test_mutants(params)
    print(f"Mutation score: {mutation_score}")
