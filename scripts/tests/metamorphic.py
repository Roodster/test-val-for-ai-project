import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import onnxruntime as rt
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

def test_metamorphic(data, feature_name : str, is_Fraud: bool ,value_from : int, value_to: int, model_path : str):
    data_modified = data.copy() 
    is_checked = 1 if is_Fraud else 0
    test_data= data_modified.loc[data_modified['checked'] == is_checked]
    test_data = test_data.loc[test_data[feature_name] == value_from]
    data_wout = test_data.copy()
    y_wout = data_wout['checked']
    X_wout= data_wout.drop(['checked'], axis=1)
    X_wout = X_wout.astype(np.float32)
    test_data[feature_name] = value_to
    y_test = test_data['checked']
    X_test= test_data.drop(['checked'], axis=1)
    X_test = X_test.astype(np.float32)
    session = rt.InferenceSession(model_path)
        
    try:
        y_pred_onnx1 =  session.run(None, {'X': X_wout.values.astype(np.float32)})
        y_pred_onnx2 =  session.run(None, {'X': X_test.values.astype(np.float32)})
        y_pred_onnx1_np = np.array(y_pred_onnx1[0])
        y_pred_onnx2_np = np.array(y_pred_onnx2[0])
        diff_count = np.sum(y_pred_onnx1_np != y_pred_onnx2_np)
        return diff_count, len(y_pred_onnx1_np)
    except InvalidArgument:
        #print("Got error: empty test dataframe. " "Column is: ", feature_name, " combo is:", is_Fraud, value_from, value_to)
        return 0, 0
    # assert y_pred_onnx1[0].all() == y_pred_onnx2[0].all(), f'Model predictions are different. The model has bias towards {feature_name} with value {value_from} '
    # return True

def plot_metamorphic_results(pre_trained_models):
    data = []
    for model in pre_trained_models:
        # Extract the percentage (it's the 2nd tuple value) and multiply it with 100
        data.append([tup[1]*100 for tup in model.column_avg])
    
    plt.boxplot(data)
    # Add title and labels
    plt.title('Box Plot of Arrays')
    plt.xlabel('Arrays')
    plt.ylabel('Values')

    # Show the plot
    plt.show()

class MetamorphicTester:
    def __init__(self, data_path, model_path):
        self.model = model_path
        self.df = pd.read_csv(data_path)
        self.mutation_test_results = {}
        self.column_avg = None
        self.global_average = None
        
    def test(self):
        # Iterate over each column in the DataFrame and generate the mutation test combinations
        for column in self.df.columns:
            # Get unique values for the current column
            unique_values = self.df[column].unique()
            if len(unique_values) > 1:
                # We sample two random values from the unique options, and store this in the dictionary
                rand_values = np.random.choice(unique_values, size=2, replace=False)
                combos = []
                for bool in [True, False]:
                    for a in rand_values:
                        for b in rand_values:
                            combos.append([bool, a, b])
                
                self.mutation_test_results[column] = combos
        
        # Go over each combination and get how many tests failed to kill the mutation
        for column, values in self.mutation_test_results.items():
            for combo in values:
                num_differ, length = test_metamorphic(self.df, column, combo[0], combo[1], combo[2], self.model)
                combo.append(num_differ)
                combo.append(length)
    
    def analyse_test(self):
        # We can now do some basic data analysis.
        # We can compute a score on a per-column basis, getting an average across 8 tests of what percentage of mutants were failed to kill
        # We can also compute a sort of global metric. Just an average across all mutations
        column_avg = []
        all_percentages = []
        
        for column, values in self.mutation_test_results.items():
            percentages = []
            for combo in values:
                # compute percentage, and add it to a temp array to compute the average value
                if combo[4] != 0:
                    res = combo[3]/combo[4]
                    percentages.append(res)
                    all_percentages.append(res)
            column_avg.append((column, np.mean(np.array(percentages))))
        
        column_avg = sorted(column_avg, key=lambda x: x[1], reverse=True)
        
        # Save the results
        self.column_avg = column_avg
        self.global_average = np.mean(np.array(all_percentages)) * 100

# features_to_ignore = ["adres_dagen_op_adres", "afspraak_aantal_woorden", "afspraak_laatstejaar_aantal_woorden", "belemmering_dagen_financiele_problemen", 
#                       "belemmering_dagen_lichamelijke_problematiek", "belemmering_dagen_psychische_problemen", "contacten_onderwerp_overige", "contacten_onderwerp_terugbelverzoek", "contacten_onderwerp_traject", "contacten_soort_afgelopenjaar_document__uitgaand_", 
#                       "contacten_soort_document__inkomend_", "contacten_soort_document__uitgaand_", "contacten_soort_e_mail__inkomend_", "contacten_soort_e_mail__uitgaand_", "contacten_soort_telefoontje__inkomend_", "contacten_soort_telefoontje__uitgaand_", 
#                       "deelname_act_reintegratieladder_werk_re_integratie", "ontheffing_dagen_hist_mean", "ontheffing_dagen_hist_vanwege_uw_medische_omstandigheden", "persoon_leeftijd_bij_onderzoek", "persoonlijke_eigenschappen_dagen_sinds_opvoer", "persoonlijke_eigenschappen_dagen_sinds_taaleis", "persoonlijke_eigenschappen_spreektaal", "relatie_kind_leeftijd_verschil_ouder_eerste_kind", "typering_dagen_som"]
