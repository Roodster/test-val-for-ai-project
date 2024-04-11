import dice_ml
from dice_ml.utils import helpers 

class CounterFactualTester:
    
    def __init__(self):
        pass
    
    
    def run_counterfactuals(self):
        data = dice_ml.Data(dataframe=dataset, 
                   continuous_features=["persoon_leeftijd_bij_onderzoek", "adres_dagen_op_adres"],
                   outcome_name="checked")

        rf_dice = dice_ml.Model(model=model, 
                                # There exist backends for tf, torch, ...
                                backend="sklearn")

        explainer = dice_ml.Dice(data, 
                                rf_dice, 
                                # Random sampling, genetic algorithm, kd-tree,...
                                method="random")
        
        data = X_train[3:4]
        cf_examples = explainer.generate_counterfactuals(data,
                                                total_CFs=1,
                                                desired_class="opposite")
        print(cf_examples.cf_examples_list[0].final_cfs_df)

    def run