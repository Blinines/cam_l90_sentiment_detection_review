# -*- coding: utf-8 -*-
import json
from helpers import get_accuracy

with open('results_2019-10-29.json') as json_file:
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

# printing results
for t in acc.keys():
    for smoothing in data[t].keys():
        print("Type: {0}, smoothing: {1}".format(t, smoothing))
        print(acc[t][smoothing])
        print('')

