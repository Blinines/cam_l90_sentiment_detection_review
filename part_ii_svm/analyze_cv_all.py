# -*- coding: utf-8 -*-
import json
import ast
import numpy as np
import pandas as pd
from helpers.helpers_gen import p_value_permutation_test

def get_right_format(str):
    res = []
    l = [n.strip() for n in str]
    for elt in l:
       if elt in ['0', '1']:
           res.append(int(elt))
    return np.array(res)


analyze_bow = False
if analyze_bow:
    with open('experiments/SVM_BOW_results_2019_11_28_14_49_01.json', 'r') as json_file:
        data_bow = json.load(json_file)

    y_true_bow = np.array((data_bow['true_pred']))
    models_bow = sorted(list(data_bow['models_res'].keys()))
    for key, value in data_bow['models_res'].items():
        data_bow['models_res'][key] = get_right_format(value)


    types = ['unigram', 'bigram', 'joint']
    models_bow = ['feat_type_{0}_tf_idf_0_norm_False', 'feat_type_{0}_tf_idf_0_norm_True',
            'feat_type_{0}_tf_idf_1_norm_False', 'feat_type_{0}_tf_idf_1_norm_True']
    for type_mod in types:
        print('type: {0}'.format(type_mod))
        curr_models = [mod.format(type_mod) for mod in models_bow]
        for index_1, model_1 in enumerate(curr_models):
            if index_1 != 3:
                for model_2 in curr_models[index_1+1:]:
                    print('mod_1 => {0}, mod_2 => {1}, p_value => {2}'.format(model_1,
                                                                            model_2,
                                                                            p_value_permutation_test(data_bow['models_res'][model_1], data_bow['models_res'][model_2], y_true_bow)))


analyze_doc2vec = True
if analyze_doc2vec:
    with open('experiments/SVM_doc2vec_results_2019_12_02_22_29_02.json', 'r') as json_file:
        data_doc2vec = json.load(json_file)

    y_true_doc2vec = np.array((data_doc2vec['true_pred']))
    models_doc2vec = sorted(list(data_doc2vec['models_res'].keys()))
    for key, value in data_doc2vec['models_res'].items():
        data_doc2vec['models_res'][key] = get_right_format(value)

    models_doc2vec = ['dm_0_dbow_words_0', 'dm_0_dbow_words_1',
                      'dm_1_dbow_words_0', 'dm_1_dbow_words_1']
    
    print('Results for CV - Doc2Vec')
    for index_1, model_1 in enumerate(models_doc2vec):
        if index_1 != 3:
            for model_2 in models_doc2vec[index_1+1:]:
                print('mod_1 => {0}, mod_2 => {1}, p_value => {2}'.format(model_1,
                                                                          model_2,
                                                                          p_value_permutation_test(data_doc2vec['models_res'][model_1], data_doc2vec['models_res'][model_2], y_true_doc2vec)))