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
from data_model.formatted_data import X_train, y_train, X_dev, X_test_blind, y_test_blind, y_dev

X_train_, y_train_ = deepcopy(X_train), deepcopy(y_train)
X_test, y_test = deepcopy(X_test_blind), deepcopy(y_test_blind)

results = {"true_values": y_test, "res_models": {}}

# Results from NB with add-one smoothing : unigrams, bigrams, joint
NB_u = {'clf': NaiveBayes(t='unigram'), 'name': 'NB_unigram'}
NB_b = {'clf': NaiveBayes(t='bigram'), 'name': 'NB_bigram'}
NB_j = {'clf': NaiveBayes(t='joint'), 'name': 'NB_joint'}

# Results from SVM - BoW : unigrams, bigrams, joint
SVM_bow_u = {'clf': Pipeline([('svmbow', SVMBOW(t='unigram')), 
                              ('svm', SVC(gamma='scale'))]), 
             'name': 'SVM_BOW_unigram'}
SVM_bow_b = {'clf': Pipeline([('svmbow', SVMBOW(t='bigram')), 
                              ('svm', SVC(gamma='scale'))]), 
             'name': 'SVM_BOW_bigram'}
SVM_bow_j = {'clf': Pipeline([('svmbow', SVMBOW(t='joint')), 
                              ('svm', SVC(gamma='scale'))]), 
             'name': 'SVM_BOW_joint'}

# Results from SVM - Doc2Vec : unigrams only
SVM_doc2vec = {'clf': Pipeline([('doc2vec', Doc2VecModel()), 
                                ('svm', SVC(C=0.1, degree=5, gamma='scale', kernel='linear'))]), 
               'name': 'SVM_doc2vec'}


if __name__ == '__main__':
    all_clf_info = [NB_u, NB_b, NB_j, SVM_bow_u, SVM_bow_b, SVM_bow_j, SVM_doc2vec]
    for clf_info in all_clf_info:
        print("Classifier trained : {0}".format(clf_info["name"]))
        clf = clf_info['clf']
        clf.fit(X_train, y_train)
        y_predicted = list(clf.predict(X_test))
        results["res_models"][clf_info["name"]] = [int(elt) for elt in y_predicted]
    
    # Printing results
    for model in results["res_models"].keys():
        print('{0}, accuracy : {1}'.format(model, np.mean(np.array(results["true_values"])==np.array(results["res_models"][model]))))

    # Storing results into .json
    save_json = False
    if save_json:
        curr_date = str(datetime.now())
        y, mo, d, h, mi, s = tuple(curr_date[:10].split('-') + curr_date[11:19].split(':'))
        file_name_no_ext = "results/all_results_{0}_{1}_{2}_{3}_{4}_{5}".format(y, mo, d, h, mi, s)
    
        with open("{0}.json".format(file_name_no_ext),"w") as fp:
            json.dump(results, fp, default=str)