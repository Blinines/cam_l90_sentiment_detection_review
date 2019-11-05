# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from helpers_cv import folder_round_robin
from helpers_nb import create_feat_no_s
from settings import PATH_NEG_TAG, PATH_POS_TAG

class Doc2VecModel(BaseEstimator):

    def __init__(self, dm=0, vector_size=50, window=2, epochs=40, hs=0):
        self.d2v_model = None
        self.dm = dm
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.hs = hs

    def fit(self):
        # As training a model takes a long time => simply opening the saved model
        model_name = "dm_{0}_vector_size_{1}window_{2}epoch_{3}hs_{4}" \
                        .format(self.dm, self.vector_size, self.window, self.epochs, self.hs)
        self.d2v_model = Doc2Vec.load("models_svm/{0}".format(model_name))
        return self

    def transform(self, raw_documents):
        X = []
        for elt in raw_documents:
            features = create_feat_no_s(file_path=elt)
            X.append(self.d2v_model.infer_vector(features))
        return X

    def fit_transform(self, raw_documents, y=None):
        self.fit()
        return self.transform(raw_documents)




param_grid = {'doc2vec__dm': [0, 1],
              'doc2vec__vector_size': [50, 100],
              'doc2vec__window': [2, 4],
              'doc2vec__epochs': [20, 40],
              'doc2vec__hs': [0,1],
              'svm__kernel': ['linear', 'rbf'],
              'svm__C': [0.1, 1],
}

pipe_log = Pipeline([('doc2vec', Doc2VecModel()), ('svm', SVC(gamma='scale'))])

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

# fitted = log_grid.fit(X_train, y_train)

# # Best parameters
# print("Best Parameters: {}\n".format(log_grid.best_params_))
# print("Best accuracy: {}\n".format(log_grid.best_score_))
# print("Finished.")
