# -*- coding: utf-8 -*-
import smart_open
import gensim
from os import listdir
from private import PATH_PROJECT

svm_train_folder_dir = ['aclImdb/test/neg/', 'aclImdb/test/pos/', 'aclImdb/train/neg/', 'aclImdb/train/pos/']

svm_train_file_dir = []
for folder in svm_train_folder_dir:
    svm_train_file_dir += [PATH_PROJECT + folder + file_name for file_name in listdir(PATH_PROJECT + folder)]


def read_corpus(files_path, tokens_only=False):
    for file_path in files_path:
        with smart_open.open(file_path, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                tokens = gensim.utils.simple_preprocess(line)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(svm_train_file_dir))
# test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

with open("train_corpus.json", "w") as fp:
    json.dump(train_corpus, fp)