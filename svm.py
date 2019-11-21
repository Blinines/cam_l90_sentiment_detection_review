# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from os import listdir
import gensim
from gensim.models.doc2vec import Doc2Vec
from helpers.helpers_cv import folder_round_robin
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG
from helpers.helpers_bow import create_feat_no_s
from part_ii_svm.create_doc2vec_model import read_corpus
from pipeline_svm import Doc2VecModel
from gensim.models.doc2vec import Doc2Vec
from ressources.settings import PATH_PROJECT



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


train_model = True
if train_model:
    # Training model
    svm_train_folder_dir = ['aclImdb/test/neg/', 'aclImdb/test/pos/', 
                            'aclImdb/train/neg/', 'aclImdb/train/pos/', 'aclImdb/train/unsup/']
    svm_train_file_dir = []
    for folder in svm_train_folder_dir:
        svm_train_file_dir += [PATH_PROJECT + folder + file_name \
                                    for file_name in listdir(PATH_PROJECT + folder)]

    train_corpus = list(read_corpus(svm_train_file_dir))
    doc2vec_model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=100, \
                                                  window=4, epoch=20, hs=1,
                                                  alpha=0.025, min_alpha=0.0001, negative=5,
                                                  dbow_words=1, min_count=1, sample=1e-5)
    doc2vec_model.build_vocab(train_corpus)
    doc2vec_model.train(train_corpus, total_examples=doc2vec_model.corpus_count, 
                        epochs=doc2vec_model.epochs)
    
    def transform(model, raw_documents):
        X = []
        for elt in raw_documents:
            features = create_feat_no_s(file_path=elt)
            X.append(model.infer_vector(features, alpha=0.01, epochs=1000))
        return X
    
    X_train_vectors = transform(doc2vec_model, X_train)
    X_test_vectors = transform(doc2vec_model, X_test_blind)

load_model = False
if load_model:
    # Loading model
    doc2vec_model = Doc2VecModel(dm=0, vector_size=100, window=4, epochs=20, hs=1)
    doc2vec_model.fit()

    X_train_vectors = doc2vec_model.transform(X_train)
    X_test_vectors = doc2vec_model.transform(X_test_blind)


clf = svm.SVC(C=0.1, degree=3, gamma='auto', kernel='linear')
clf.fit(X_train_vectors, y_train) 
y_predicted = clf.predict(X_test_vectors)

print("Accuracy: {0}".format(np.mean(y_predicted==y_test_blind)))
