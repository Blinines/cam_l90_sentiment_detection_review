# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from gensim.models.doc2vec import Doc2Vec
from helpers.helpers_cv import folder_round_robin
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG
from helpers.helpers_nb import create_feat_no_s
from part_ii_svm.pipeline_svm import Doc2VecModel


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

X_test_blind_neg = fold_rr_neg[0]
X_test_blind_pos = fold_rr_pos[0]
X_test_blind = X_test_blind_neg + X_test_blind_pos
y_test_blind = [0]*len(X_test_blind_neg) + [1]*len(X_test_blind_pos)


# Loading model
doc2vec_model = Doc2VecModel(dm=0, vector_size=100, window=4, epochs=20, hs=1)
doc2vec_model.fit()

X_train_vectors = doc2vec_model.transform(X_train)
X_test_vectors = doc2vec_model.transform(X_test_blind)


clf = svm.SVC(C=0.1, degree=3, gamma='auto', kernel='linear')
clf.fit(X_train_vectors, y_train) 
y_predicted = clf.predict(X_test_vectors)

print("Accuracy: {0}".format(np.mean(y_predicted==y_test_blind)))
