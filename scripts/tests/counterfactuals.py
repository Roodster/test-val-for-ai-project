import dice_ml

def setup_cf(data, model):
    """
    This function sets up a counterfactual explanation object.

    Args:
    data: A pandas dataframe containing the data used for explanation.
    model: A machine learning model used for generating counterfactuals.

    Returns:
    A dice_ml.Dice object representing the counterfactual explainer.
    """
  
    data = dice_ml.Data(dataframe=data, 
                   continuous_features=["persoon_leeftijd_bij_onderzoek", "adres_dagen_op_adres"],
                   outcome_name="checked")

    rf_dice = dice_ml.Model(model=model, 
                            # There exist backends for tf, torch, ...
                            backend="sklearn")

    explainer = dice_ml.Dice(data, 
                            rf_dice, 
                            # Random sampling, genetic algorithm, kd-tree,...
                            method="random")
    
    return explainer
        
def run_cf(data_sample, explainer):
    """
    This function generates counterfactual examples for a given data sample.

    Args:
    data_sample: A pandas dataframe representing the data sample for which to generate counterfactuals.
    explainer: A dice_ml.Dice object representing the counterfactual explainer.

    Returns:
    A dice_ml.Explanation object containing the generated counterfactual examples.
    """
    cf_examples = explainer.generate_counterfactuals(data_sample,
                                            total_CFs=1,
                                            desired_class="opposite")
    return cf_examples

def generate_dict(data):
    """
    This function creates a dictionary to store counterfactual analysis results.

    Args:
    data: A pandas dataframe containing the data used for analysis.

    Returns:
    A dictionary where keys are column names and values are dictionaries containing
    possible values for that column and their corresponding counts and desired class counts.
    """
    results = {}
    for i in data.columns.tolist():
        possible_values = data[i].unique()
        temp = {}
        for j in possible_values:
            temp[str(j)] = {'count':0, 'desired_class_0':0, 'desired_class_1': 0}
        results[i] = temp
    return results

def analyse_cf_test(data, cf_examples):
    """
    This function analyzes counterfactual examples and populates a results dictionary.

    Args:
    data: A pandas dataframe containing the original data.
    cf_examples: A dice_ml.Explanation object containing the counterfactual examples.

    Returns:
    A dictionary containing analysis results. Keys are column names, and values are
    dictionaries mapping possible values to their counts and desired class counts.
    """
  
    results_dict = generate_dict(data)
    for idx, cf_df in enumerate(cf_examples.cf_examples_list):
        cf = cf_df.final_cfs_df.iloc[0]
        ori = data.iloc[idx]
        assert cf_df.desired_class == cf_df.new_outcome

        for (col1, ori_val), (col2, cf_val) in zip(ori.iteritems(), cf.iteritems()):
            if ori_val == cf_val: continue
            if col1 == "checked": continue
            assert col1 == col2
            
            dict = results_dict[col1].get(str(cf_val), {'count':0, 'desired_class_0':0, 'desired_class_1': 0})
            new_dict = {'count': dict['count'] + 1}
            new_dict['desired_class_1'] = (dict['desired_class_1'] + 1) if (cf_df.desired_class == 1) else dict['desired_class_1']
            new_dict['desired_class_0'] = (dict['desired_class_0'] + 1) if (cf_df.desired_class == 0) else dict['desired_class_0']
            results_dict[col1][str(cf_val)] = new_dict
            
    return results_dict

def return_sorted_cf_results(cf_test_dict):
    """
    This function sorts the counterfactual analysis results by count.

    Args:
    cf_test_dict: A dictionary containing counterfactual analysis results.

    Returns:
    A list of tuples containing sorted analysis results for each column and value.
    """
    temp = []
    for col_name, value_dicts in cf_test_dict.items():
        for value, count_dict in value_dicts.items():
            temp.append((col_name, value, "count", count_dict['count'], "desired_class_0", count_dict['desired_class_0'], "desired_class_1", 
                         count_dict["desired_class_1"]))
    
    return sorted(temp, key=lambda x: x[3], reverse=True)

def cf_pipeline(dataset, test_data, audit_model):
    """
    This function performs the entire counterfactual analysis pipeline.

    Args:
        dataset: A pandas dataframe containing the entire dataset.
        test_data: A pandas dataframe containing the data samples for which to generate counterfactuals.
        audit_model: A machine learning model used for generating counterfactuals.

    Returns:
        A tuple containing:
        - explainer_model: A dice_ml.Dice object representing the counterfactual explainer.
        - cf_examples_model: A dice_ml.Explanation object containing the generated counterfactuals.
        - model_results_dict: A dictionary containing analysis results.
        - model_results_sorted: A list of tuples containing sorted analysis results for each column and value.
    """
  
    explainer_model = setup_cf(dataset, model=audit_model)
    cf_examples_model = run_cf(test_data, explainer=explainer_model)
    model_results_dict = analyse_cf_test(test_data, cf_examples=cf_examples_model)
    model_results_sorted = return_sorted_cf_results(cf_test_dict=model_results_dict)
    return explainer_model, cf_examples_model, model_results_dict, model_results_sorted
