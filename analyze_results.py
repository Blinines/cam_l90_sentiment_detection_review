# -*- coding: utf-8 -*-
import json
import numpy as np
from helpers import get_accuracy, sign_test, p_value, get_variance

with open('results_nb.json') as json_file:
    data = json.load(json_file)

# getting accuracy per testing fold and averaged accuracy
acc = {}
for t in data.keys():
    acc[t] = {}
    for smoothing in data[t].keys():
        acc[t][smoothing] = {'l': [], 'val': None, 'randomed_l': []}
        for test_fold in data[t][smoothing].keys():
            acc[t][smoothing]['l'].append(get_accuracy(data[t][smoothing][test_fold]))
            #acc[t][smoothing]['randomed_l'].append(data[t][smoothing]['random'])
        acc[t][smoothing]['val'] = np.mean(acc[t][smoothing]['l'])
        acc[t][smoothing]['variance'] = get_variance(acc[t][smoothing]['l'])
        #acc[t][smoothing]['randomed'] = np.mean(acc[t][smoothing]['randomed_l'])


def concat_prediction(data):
    concat = {}
    for t in data.keys():
        for smoothing in data[t].keys():
            curr_res = data[t][smoothing]
            new_name = '{0}_{1}'.format(t, smoothing)
            concat[new_name] = {}
            for val in ['NEG', 'POS']:
                concat[new_name][val] = []
                for key in curr_res.keys():
                    concat[new_name][val] += curr_res[key][val]
    return concat



# printing results
print_result_acc = True
if print_result_acc:
    for t in acc.keys():
        for smoothing in data[t].keys():
            print("Type: {0}, smoothing: {1}".format(t, smoothing))
            print(acc[t][smoothing])
            print('')

print_sign_test = True
if print_sign_test:
    concat = concat_prediction(data)
    models = sorted(concat.keys())
    for i in range(len(models)-1):
        model_1 = models[i]
        for model_2 in models[i+1:]:
                numbers = sign_test(concat[model_1], concat[model_2])
                p_val = p_value(numbers=numbers, q=0.5)
                print("Sign test : 1 => {0}, 2 => {1}, p value is {2}, numbers are {3}".format(model_1, model_2, p_val, numbers))
