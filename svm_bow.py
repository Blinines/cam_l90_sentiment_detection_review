# -*- coding: utf-8 -*-
from collections import Counter
from scipy import sparse
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from helpers.helpers_cv import folder_round_robin
from helpers.helpers_nb import create_feat_no_s
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG


def map_word_to_id(train_files_path):
        word_to_id = Counter()
        max_id = 0
        for train_file_path in train_files_path:
            feat_set = set(create_feat_no_s(file_path=train_file_path))
            for feat in feat_set:
                if feat not in word_to_id.keys(): #adding new mapping word => id
                    word_to_id[feat] = max_id
                    max_id += 1
        return word_to_id


def create_vect_for_doc(file_path, word_to_id):
    x = np.zeros((1, len(word_to_id.keys())))
    features = create_feat_no_s(file_path=file_path)
    for feat in features:
        if feat in word_to_id.keys():
            x[0, word_to_id[feat]] += 1
    return x


class SVMBOW():

    def __init__(self):
        self.bow = None
    
    def fit(self, raw_documents):
        self.bow = map_word_to_id(train_files_path=raw_documents)
        return self
    
    def transform(self, raw_documents):
        X = np.zeros((len(raw_documents), len(self.bow.keys())))
        for index, doc in enumerate(raw_documents):
            X[index, :] = create_vect_for_doc(file_path=doc, word_to_id=self.bow)
        return sparse.coo_matrix(X)
    
    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)


param_grid = {'svm__kernel': ['linear', 'rbf', 'poly'],
              'svm__C': [0.1, 0.5, 1, 10],
              'svm__degree': [2, 3, 5],
              'svm__gamma': ['scale', 'auto']}


grid_search = False
if grid_search:
    pipe_log = Pipeline([('svmbow', SVMBOW()), ('svm', SVC(gamma='scale'))])

    log_grid = GridSearchCV(pipe_log, 
                            param_grid=param_grid,
                            scoring="accuracy",
                            verbose=3,
                            cv=10,
                            n_jobs=-1)


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

    fitted = log_grid.fit(X_train, y_train)

    # Best parameters
    print("Best Parameters: {}\n".format(log_grid.best_params_))
    print("Best accuracy: {}\n".format(log_grid.best_score_))
    print("Finished.")