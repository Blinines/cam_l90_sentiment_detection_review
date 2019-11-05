# -*- coding: utf-8 -*-
import json
from helpers import get_accuracy, sign_test, p_value

with open('results_2019-11-04.json') as json_file:
    data = json.load(json_file)

# getting accuracy per testing fold and averaged accuracy
acc = {}
for t in data.keys():
    acc[t] = {}
    for smoothing in data[t].keys():
        acc[t][smoothing] = {'l': [], 'val': None}
        for test_fold in data[t][smoothing].keys():
            acc[t][smoothing]['l'].append(get_accuracy(data[t][smoothing][test_fold]))
        acc[t][smoothing]['val'] = float(sum(acc[t][smoothing]['l'])) / len(acc[t][smoothing]['l'])


def concat_prediction(data):
    concat = {}
    for t in data.keys():
        for smoothing in data[t].keys():
            curr_res = data[t][smoothing]
            new_name = '{0}_{1}'.format(t, smoothing)
            concat[new_name] = {}
            for val in ['NEG', 'POS']:
                concat[new_name][val] = []
                for key in ["0"]:
                # for key in curr_res.keys():
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

print_sign_test = False
if print_sign_test:
    concat = concat_prediction(data)
    models = sorted(concat.keys())
    for model_1 in models:
        for model_2 in models:
            if model_2 != model_1:
                p_val = p_value(numbers=sign_test(concat[model_1], concat[model_2]), q=0.5)
                print("Sign test : 1 => {0}, 2 => {1}, p value is {2}".format(model_1, model_2, p_val))
