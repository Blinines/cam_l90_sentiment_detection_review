# -*- coding: utf-8 -*-
from copy import deepcopy
from formatted_data import X_train, y_train, X_test_blind, y_test_blind

X_train_, y_train_ = deepcopy(X_train), deepcopy(y_train)
X_test, y_test = deepcopy(X_test_blind), deepcopy(y_test_blind)

results = {"true_values": y_test, "res_models": {}}

# Results from NB with add-one smoothing : unigrams, bigrams, joint
# Results from SVM - BoW : unigrams, bigrams, joint
# Results from SVM - Doc2Vec : unigrams only