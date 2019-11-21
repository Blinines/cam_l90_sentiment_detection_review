# -*- coding: utf-8 -*-
from collections import Counter
from scipy import sparse
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from helpers.helpers_cv import folder_round_robin
from helpers.helpers_bow import create_feat_no_s, create_bow
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG, FREQ_CUTOFF_UNIGRAM, FREQ_CUTOFF_BIGRAM


def map_word_to_id(bow_count):
    word_to_id = Counter()
    features = list(bow_count.keys())
    for index, feat in enumerate(features):
        word_to_id[feat] = index
    return word_to_id


def create_vect_for_doc(file_path, word_to_id):
    x = np.zeros((1, len(word_to_id.keys())))
    features = create_feat_no_s(file_path=file_path)
    for feat in features:
        if feat in word_to_id.keys():
            x[0, word_to_id[feat]] += 1
    return x


class SVMBOW:

    def __init__(self, t, freq_cutoff, tf_idf=0):
        # t is type, in ['unigram', 'bigram', 'joint']
        # freq_cutoff = {'unigram': 1, 'bigram': 4} for example
        self.bow = None
        self.t = t
        self.type_to_calc = {'unigram': [1], 'bigram': [2], 'joint': [1, 2]}
        self.freq_cutoff = freq_cutoff
        self.tf_idf = tf_idf
    
    def fit(self, raw_documents):
        bow_count = {}
        for nb in self.type_to_calc[self.t]:
            curr_bow= create_bow(raw_documents, self.freq_cutoff[nb], nb)[0]
            bow_count.update(curr_bow)

        self.bow = map_word_to_id(bow_count=bow_count)
        return self
    
    def transform(self, raw_documents):
        X = np.zeros((len(raw_documents), len(self.bow.keys())))
        for index, doc in enumerate(raw_documents):
            X[index, :] = create_vect_for_doc(file_path=doc, word_to_id=self.bow)
        
        if self.tf_idf:
            tf_transformer = TfidfTransformer(use_idf=False).fit(X)
            X = tf_transformer.transform(X)

        return sparse.coo_matrix(X)

    
    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)


param_grid = {'svm__kernel': ['linear', 'rbf', 'poly'],
              'svm__C': [0.1, 0.5, 1, 10, 100],
              #'svm__degree': [2, 3, 5],
              #'svm__gamma': ['scale', 'auto']
              'svm__gamma': [1, 0.1, 0.01, 10]}


grid_search = False
if grid_search:
    # Preparing parameters
    FREQ_CUTOFF = {1: FREQ_CUTOFF_UNIGRAM, 2: FREQ_CUTOFF_BIGRAM} 

    fold_rr_neg = folder_round_robin(files_path=PATH_NEG_TAG, mod=10)
    X_train_neg = []
    for i in range(1, 10):
        X_train_neg += fold_rr_neg[i]

    fold_rr_pos = folder_round_robin(files_path=PATH_POS_TAG, mod=10)
    X_train_pos = []
    for i in range(1, 10):
        X_train_pos += fold_rr_pos[i]

    X_train = X_train_neg + X_train_pos
    y_train = [0]*len(X_train_neg) + [1]*len(X_train_pos)

    res = {}
    for t in ['unigram', 'bigram', 'joint']:
        for tf_idf in [0, 1]:
            pipe_log = Pipeline([('svmbow', SVMBOW(t=t, freq_cutoff=FREQ_CUTOFF, tf_idf=tf_idf)), 
                                ('svm', SVC(gamma='scale'))])

            log_grid = GridSearchCV(pipe_log, 
                                    param_grid=param_grid,
                                    scoring="accuracy",
                                    verbose=3,
                                    cv=2,
                                    n_jobs=-1)


            fitted = log_grid.fit(X_train, y_train)
            res[t] = log_grid
            print("Results for Gridsearch: features => {0}, tf_idf => {1}\n".format(t, tf_idf))
            print("Best Parameters: {}\n".format(log_grid.best_params_))
            print("Best accuracy: {}\n".format(log_grid.best_score_))
            print("Finished.")


    # Best parameters
    for t in res.keys():
        print("Results for Gridsearch: {}\n".format(t))
        print("Best Parameters: {}\n".format(res[t].best_params_))
        print("Best accuracy: {}\n".format(res[t].best_score_))
        print("Finished.")