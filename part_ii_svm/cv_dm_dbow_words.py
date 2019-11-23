# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from part_ii_svm.doc_embeddings import Doc2VecModel
from helpers.helpers_gen import p_value_permutation_test
from data_model.formatted_data import X_train, y_train, X_dev, y_dev


cv = False
if cv:
    # DBOW vs DM architecture
    print("DBOW vs DM architecture")
    param_grid = { 'doc2vec__dm': [0, 1] }  # if 1 dm architecture
    pipe_log = Pipeline([('doc2vec', Doc2VecModel()), ('svm', SVC(gamma='scale'))])
    log_grid = GridSearchCV(pipe_log, param_grid=param_grid, scoring="accuracy",
                            verbose=3, cv=5, n_jobs=-1)
    X_train_, y_train_ = deepcopy(X_train), deepcopy(y_train)
    fitted = log_grid.fit(X_train_, y_train_)

    print("Best Parameters: {}".format(log_grid.best_params_))
    print("Best accuracy: {}\n".format(log_grid.best_score_))


    # Training jointly word embedding or not
    print("Training jointly word embedding or not")
    param_grid = { 'doc2vec__dbow_words': [0, 1] }  # if 1 word embeddings trained jointly
    pipe_log = Pipeline([('doc2vec', Doc2VecModel()), ('svm', SVC(gamma='scale'))])
    log_grid = GridSearchCV(pipe_log, param_grid=param_grid, scoring="accuracy",
                            verbose=3, cv=5, n_jobs=-1)
    X_train_, y_train_ = deepcopy(X_train), deepcopy(y_train)
    fitted = log_grid.fit(X_train_, y_train_)

    print("Best Parameters: {}".format(log_grid.best_params_))
    print("Best accuracy: {}\n".format(log_grid.best_score_))


dev_test = True
if dev_test:
    print("DBOW vs DM architecture")
    pipe_clf_dbow = Pipeline([('doc2vec', Doc2VecModel(dm=0)), ('svm', SVC(gamma='scale'))])
    pipe_clf_dm = Pipeline([('doc2vec', Doc2VecModel(dm=1)), ('svm', SVC(gamma='scale'))])
    X_train_, y_train_ = deepcopy(X_train), deepcopy(y_train)

    pipe_clf_dbow.fit(X_train_, y_train_)
    pipe_clf_dm.fit(X_train_, y_train_)
    y_dbow = pipe_clf_dbow.predict(X_dev)
    y_dm = pipe_clf_dm.predict(X_dev)
    
    print("DBOW accuracy: {0}".format(np.mean(y_dbow==y_dev)))
    print("DM accuracy: {0}".format(np.mean(y_dm==y_dev)))
    print("p value: {0} \n".format(p_value_permutation_test(y_dbow, y_dm, y_dev)))


    print("Training word embeddings jointly or not")
    pipe_clf_no_train_joint = Pipeline([('doc2vec', Doc2VecModel(dbow_words=0)), ('svm', SVC(gamma='scale'))])
    pipe_clf_train_join = Pipeline([('doc2vec', Doc2VecModel(dbow_words=1)), ('svm', SVC(gamma='scale'))])
    X_train_, y_train_ = deepcopy(X_train), deepcopy(y_train)

    pipe_clf_no_train_joint.fit(X_train_, y_train_)
    pipe_clf_train_join.fit(X_train_, y_train_)
    y_no_train_joint = pipe_clf_no_train_joint.predict(X_dev)
    y_train_joint = pipe_clf_train_join.predict(X_dev)
    
    print("No joint training: {0}".format(np.mean(y_no_train_joint==y_dev)))
    print("Joint training: {0}".format(np.mean(y_train_joint==y_dev)))
    print("p value: {0} \n".format(p_value_permutation_test(y_no_train_joint, y_train_joint, y_dev)))
