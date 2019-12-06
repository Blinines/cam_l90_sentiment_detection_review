# -*- coding: utf-8 -*-
import json
import numpy as np
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from datetime import datetime
from part_i_naive_bayes.naive_bayes import NaiveBayes
from part_ii_svm.doc_embeddings import Doc2VecModel, SVMBOW
from ressources.settings import FREQ_CUTOFF
from helpers.helpers_gen import p_value_permutation_test
from data_model.formatted_data import X_train, y_train, X_dev, X_test_blind, y_test_blind, y_dev

X_train_, y_train_ = deepcopy(X_train+X_dev), deepcopy(y_train+y_dev)
X_test, y_test = deepcopy(X_test_blind), deepcopy(y_test_blind)

results = {"true_values": np.array(y_test), "res_models": {}}

# Results from NB with add-one smoothing : unigrams, bigrams, joint
NB_u = {'clf': NaiveBayes(t='unigram'), 'name': 'NB_unigram'}
NB_b = {'clf': NaiveBayes(t='bigram'), 'name': 'NB_bigram'}
NB_j = {'clf': NaiveBayes(t='joint'), 'name': 'NB_joint'}

# Results from SVM - BoW : unigrams, bigrams, joint
SVM_bow_u = {'clf': Pipeline([('svmbow', SVMBOW(t='unigram')), 
                              ('svm', SVC(gamma=10, kernel='poly', C=0.1, degree=2))]), 
             'name': 'SVM_BOW_unigram'}
SVM_bow_b = {'clf': Pipeline([('svmbow', SVMBOW(t='bigram')), 
                              ('svm', SVC(gamma='scale', kernel='linear', C=10, degree=2))]), 
             'name': 'SVM_BOW_bigram'}
SVM_bow_j = {'clf': Pipeline([('svmbow', SVMBOW(t='joint')), 
                              ('svm', SVC(gamma='scale', kernel='linear', C=10, degree=2))]), 
             'name': 'SVM_BOW_joint'}

# Results from SVM - Doc2Vec : unigrams only
SVM_doc2vec = {'clf': Pipeline([('doc2vec', Doc2VecModel()), 
                                ('svm', SVC(C=0.1, degree=3, gamma='scale', kernel='linear'))]), 
               'name': 'SVM_doc2vec'}


if __name__ == '__main__':
    all_clf_info = [NB_u, NB_b, NB_j, SVM_bow_u, SVM_bow_b, SVM_bow_j, SVM_doc2vec]
    for clf_info in all_clf_info:
        print("Classifier trained : {0}".format(clf_info["name"]))
        clf = clf_info['clf']
        clf.fit(X_train, y_train)
        y_predicted = list(clf.predict(X_test))
        results["res_models"][clf_info["name"]] = np.array([int(elt) for elt in y_predicted])
    
    # Printing results
    for model in results["res_models"].keys():
        print('{0}, accuracy : {1}'.format(model, np.mean(np.array(results["true_values"])==np.array(results["res_models"][model]))))

    print("===========")
    print(results)
    print("===========")
    models = list(results["res_models"].keys())
    nb_models = len(models)
    for index_1, model_1 in enumerate(models):
        if index_1 != nb_models-1:
            for model_2 in models[index_1+1:]:
                print('mod_1 => {0}, mod_2 => {1}, p_value => {2}'.format(model_1,
                                                                          model_2,
                                                                          p_value_permutation_test(results['res_models'][model_1], results['res_models'][model_2], y_test)))

    # Storing results into .json
    save_json = True
    if save_json:
        curr_date = str(datetime.now())
        y, mo, d, h, mi, s = tuple(curr_date[:10].split('-') + curr_date[11:19].split(':'))
        file_name_no_ext = "results/all_results_{0}_{1}_{2}_{3}_{4}_{5}".format(y, mo, d, h, mi, s)
    
        with open("{0}.json".format(file_name_no_ext),"w") as fp:
            json.dump(results, fp, default=str)