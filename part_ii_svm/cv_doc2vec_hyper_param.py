# -*- coding: utf-8 -*-
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from part_ii_svm.doc_embeddings import Doc2VecModel
from data_model.formatted_data import X_train, y_train

param_grid = {'doc2vec__dm': [0],  # [0, 1]
              'doc2vec__vector_size': [100],  # [50, 100]
              'doc2vec__window': [10, 15],  # [2, 4, 10, 15]
              'doc2vec__epochs': [20],  # [20, 40]
              'doc2vec__hs': [1],  # [0,1]
              'doc2vec__dbow_words': [0, 1],  # [0, 1]
              'doc2vec__alpha_infer': [None],  # [None, 0.01]
              'doc2vec__epochs_infer': [None],  # [None, 500, 1000]
              'svm__kernel': ['linear', 'rbf', 'poly'],  # ['linear', 'rbf', 'poly']
              'svm__C': [0.1, 1, 10],  # [0.1, 0.5, 1, 10]
              'svm__degree': [2, 3, 5],  # [2, 3, 5]
              'svm__gamma': ['scale', 'auto'],  # ['scale', 'auto']

}


pipe_log = Pipeline([('doc2vec', Doc2VecModel()), ('svm', SVC(gamma='scale'))])

log_grid = GridSearchCV(pipe_log, 
                        param_grid=param_grid,
                        scoring="accuracy",
                        verbose=3,
                        cv=3,
                        n_jobs=-1)

X_train_ = deepcopy(X_train)
y_train_ = deepcopy(y_train)
fitted = log_grid.fit(X_train, y_train_)

# Best parameters
print("Best Parameters: {}\n".format(log_grid.best_params_))
print("Best accuracy: {}\n".format(log_grid.best_score_))
print("Finished.")
