# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from helpers.helpers_cv import folder_round_robin
from data_model.formatted_data import X_train, y_train
from part_ii_svm.doc_embeddings import SVMBOW
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG, FREQ_CUTOFF



param_grid = {'svm__kernel': ['linear', 'rbf', 'poly'],
              'svm__C': [0.1, 1, 10],
              'svm__degree': [2, 3, 5],
              'svm__gamma': ['scale', 'auto'],
              #'svm__gamma': [1, 0.1, 0.01, 10]
            }


# Preparing parameters

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
        print("Results for Gridsearch: features => {0}, tf_idf => {1}\n".format(t, tf_idf))
        print("Best Parameters: {}\n".format(log_grid.best_params_))
        print("Best accuracy: {}\n".format(log_grid.best_score_))
        print("Finished.")
