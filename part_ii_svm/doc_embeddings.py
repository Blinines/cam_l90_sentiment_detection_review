# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
from collections import Counter
from os import listdir
from gensim.models.doc2vec import Doc2Vec
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from helpers.helpers_bow import create_feat_no_s, create_bow
from part_ii_svm.create_doc2vec_model import read_corpus, train_save
from ressources.settings import SVM_TRAIN_FILE_DIR


class Doc2VecModel(BaseEstimator):

    def __init__(self, dm=0, vector_size=100, window=10, epochs=20, hs=1, dbow_words=0,
                 alpha_infer=None, epochs_infer=None):
        self.d2v_model = None
        self.dm = dm
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.hs = hs
        self.dbow_words = dbow_words
        self.alpha_infer = 0.025 if alpha_infer is None else alpha_infer
        self.epochs_infer = self.epochs if epochs_infer is None else epochs_infer

    def fit(self):
        # As training a model takes a long time => some models are already saved
        # If saved => will simply open the model
        # Other : will train the model => takes more time
        model_name = "dm_{0}_vector_size_{1}_window_{2}_epoch_{3}_hs_{4}_dbow_words_{5}" \
                        .format(self.dm, self.vector_size, self.window, 
                                self.epochs, self.hs, self.dbow_words)
        if model_name not in listdir("part_ii_svm/models_svm/"):
            train_corpus = list(read_corpus(SVM_TRAIN_FILE_DIR))
            params = [[self.dm], [self.vector_size], [self.window], 
                      [self.epochs], [self.hs], [self.dbow_words]]
            train_save(params, train_corpus, write=False)

        self.d2v_model = Doc2Vec.load("part_ii_svm/models_svm/{0}".format(model_name))
        return self

    def transform(self, raw_documents):
        X = []
        for elt in raw_documents:
            features = create_feat_no_s(file_path=elt)
            X.append(self.d2v_model.infer_vector(features, alpha=self.alpha_infer, 
                                                 epochs=self.epochs_infer))
        return X

    def fit_transform(self, raw_documents, y=None):
        self.fit()
        return self.transform(raw_documents)


class SVMBOW:

    def __init__(self, t, freq_cutoff, tf_idf=0):
        # t is type, in ['unigram', 'bigram', 'joint']
        # freq_cutoff = {'unigram': 1, 'bigram': 4} for example
        self.bow = None
        self.t = t
        self.type_to_calc = {'unigram': [1], 'bigram': [2], 'joint': [1, 2]}
        self.freq_cutoff = freq_cutoff
        self.tf_idf = tf_idf
    
    def map_word_to_id(self, bow_count):
        word_to_id = Counter()
        features = list(bow_count.keys())
        for index, feat in enumerate(features):
            word_to_id[feat] = index
        return word_to_id
    
    def create_vect_for_doc(self, file_path, word_to_id):
        x = np.zeros((1, len(word_to_id.keys())))
        features = create_feat_no_s(file_path=file_path)
        for feat in features:
            if feat in word_to_id.keys():
                x[0, word_to_id[feat]] += 1
        return x
        
    def fit(self, raw_documents):
        bow_count = {}
        for nb in self.type_to_calc[self.t]:
            curr_bow= create_bow(raw_documents, self.freq_cutoff[nb], nb)[0]
            bow_count.update(curr_bow)

        self.bow = self.map_word_to_id(bow_count=bow_count)
        return self
    
    def transform(self, raw_documents):
        X = np.zeros((len(raw_documents), len(self.bow.keys())))
        for index, doc in enumerate(raw_documents):
            X[index, :] = self.create_vect_for_doc(file_path=doc, word_to_id=self.bow)
        
        if self.tf_idf:
            tf_transformer = TfidfTransformer(use_idf=False).fit(X)
            X = tf_transformer.transform(X)

        return sparse.coo_matrix(X)

    
    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)
