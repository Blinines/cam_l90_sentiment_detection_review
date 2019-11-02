# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec
from sklearn.base import BaseEstimator
from helpers_nb import create_feat_with_s, create_feat_no_s

class Doc2VecModel(BaseEstimator):

    def __init__(self, dm, vector_size, window, epochs, hs):
        self.d2v_model = None
        self.dm = dm
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.hs = hs

    def fit(self):
        # As training a model takes a long time => simply opening the saved model
        model_name = "models_svm/dm_{0}_vector_size_{1}window_{2}epoch_{3}hs_{4}" \
                        .format(self.dm, self.vector_size, self.window, self.epoch, self.hs)
        self.d2v_model = Doc2Vec.load("models_svm/{0}".format(model_name))
        return self

    def transform(self, raw_documents):
        X = []
        for elt in raw_documents:
            features = create_feat_no_s(feat_with_s=create_feat_with_s(file_path=elt))
            X.append(self.d2v_model.infer_vector(features))
        return X

    def fit_transform(self, raw_documents, y=None):
        self.fit()
        return self.transform(raw_documents)


