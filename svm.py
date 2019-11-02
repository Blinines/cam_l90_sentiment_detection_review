# -*- coding: utf-8 -*-
from sklearn import svm
from gensim.models.doc2vec import Doc2Vec
from helpers_cv import sep_train_test
from settings import PATH_NEG_TAG, PATH_POS_TAG
from helpers_nb import create_feat_with_s, create_feat_no_s

X_train_neg, X_test_blind_neg = sep_train_test(files_path=PATH_NEG_TAG, top_value_train=900)
X_train_pos, X_test_blind_pos = sep_train_test(files_path=PATH_POS_TAG, top_value_train=900)

X_train = X_train_neg + X_train_pos
y_train = [0]*len(X_train_neg) + [1]*len(X_train_pos)
X_test_blind = X_test_blind_neg + X_test_blind_pos
y_test_blind = [0]*len(X_test_blind_neg) + [1]*len(X_test_blind_pos)


# Loading model
model = Doc2Vec.load("models_svm/first_model")

# test_corpus = list(read_corpus(X_train, tokens_only=True))
# blind_corpus = list(read_corpus(X_test_blind, tokens_only=True))

X_train_vectors = []
for elt in X_train:
    features = create_feat_no_s(feat_with_s=create_feat_with_s(file_path=elt))
    X_train_vectors.append(model.infer_vector(features))


X_test_blind_vectors = []
for elt in X_test_blind:
    features = create_feat_no_s(feat_with_s=create_feat_with_s(file_path=elt))
    X_test_blind_vectors.append(model.infer_vector(features))

clf = svm.SVC(gamma='scale')
clf.fit(X_train_vectors, y_train) 
y_predicted = clf.predict(X_test_blind_vectors)

correct = 0
for index, elt in enumerate(y_predicted):
    if elt == y_test_blind[index]:
        correct += 1
print("Accuracy: {0}".format(float(correct)/len(y_predicted)))




