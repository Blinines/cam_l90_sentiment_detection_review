# -*- coding: utf-8 -*-
import json
import numpy as np
from helpers.helpers_gen import sign_test, p_value_sign_test, get_variance

# getting accuracy per testing fold and averaged accuracy
def get_acc_details(data):
    acc = {t: {} for t in data.keys()}
    for t in data.keys():
        for smoothing in data[t].keys():
            acc[t][smoothing] = {}
            for feat_type in data[t][smoothing].keys():
                acc[t][smoothing][feat_type] = {'l': []}
                for test_fold in data[t][smoothing][feat_type].keys():
                    if test_fold != "avg_score":
                        acc[t][smoothing][feat_type]['l'].append(data[t][smoothing][feat_type][test_fold]["score"])
                acc[t][smoothing][feat_type]['val'] = data[t][smoothing][feat_type]["avg_score"]
                acc[t][smoothing][feat_type]['variance'] = get_variance(acc[t][smoothing][feat_type]['l'])
    return acc


def concat_prediction(data):
    '''Concatenate all predictions from all cross validated folders'''
    concat = {}
    for t in data.keys():
        for smoothing in data[t].keys():
            for feat_type in data[t][smoothing].keys():
                curr_res = data[t][smoothing][feat_type]
                new_name = '{0}_{1}_{2}'.format(t, smoothing, feat_type)
                concat[new_name] = []

                for key in curr_res.keys():
                    if key != 'avg_score':
                        concat[new_name] += curr_res[key]['predicted']
    return concat


def create_true_values(data):
    ''' Retrieve all true values '''
    y_true = []
    
    t = list(data.keys())[0]
    smoothing = list(data[t].keys())[0]
    feat_type = list(data[t][smoothing].keys())[0]

    res = data[t][smoothing][feat_type]
    for key in res:
        if key != 'avg_score':
            y_true += res[key]['true_values']
    return [int(nb) for nb in y_true]


def analyze_results(json_path, print_result_acc=True, print_sign_test=True):
    with open(json_path) as json_file:
        data = json.load(json_file)
    
    if print_result_acc:
        acc = get_acc_details(data=data)
        for t in acc.keys():
            for smoothing in data[t].keys():
                for feat_type in data[t][smoothing].keys():
                    print("Type: {0}, smoothing: {1}, feature type: {2}".format(t, smoothing, feat_type))
                    print(acc[t][smoothing][feat_type])
                    print('')
    
    if print_sign_test:
        concat = concat_prediction(data)
        y_true = create_true_values(data)
        models = sorted(concat.keys())
        for i in range(len(models)-1):
            model_1 = models[i]
            for model_2 in models[i+1:]:
                    numbers = sign_test(y_1=concat[model_1], y_2=concat[model_2], y_true=y_true)
                    p_val = p_value_sign_test(numbers=numbers, q=0.5)
                    print("Sign test : 1 => {0}, 2 => {1}, p value is {2}, numbers are {3}".format(model_1, model_2, p_val, numbers))
        
    return


# if __name__ == '__main__':
#     json_path = '../results/NB_results_2019_11_13_22_03_39.json'
#     analyze_results(json_path)