# -*- coding: utf-8 -*-
import smart_open
import gensim
import itertools
import collections
from gensim.models.doc2vec import Doc2Vec
from os import listdir
from datetime import datetime
from ressources.settings import PATH_PROJECT, SVM_TRAIN_FILE_DIR


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


def train_save(params, train_corpus, write=True):  
    ''' Training and saving all possible models that were not saved '''
    if write: f= open("train_doc2vec_model_{0}.txt".format(str(datetime.now())[:10]),"w+")
    existing_model = listdir('models_svm/')

    for param in itertools.product(*params):
        dm, vector_size, window, epoch, hs, dbow_words = param[0], param[1], param[2], param[3], param[4], param[5]
        if write: f.write("dm: {0}, vector_size: {1}, window: {2}, epoch: {3}, hs: {4}, dbow_words: {5} \n" \
                          .format(dm, vector_size, window, epoch, hs, dbow_words))
        print("Model : dm: {0}, vector_size: {1}, window: {2}, epoch: {3}, hs: {4}, dbow_words: {5}".format(dm, vector_size, window, epoch, hs, dbow_words))
        
        name_model = "dm_{0}_vector_size_{1}_window_{2}_epoch_{3}_hs_{4}_dbow_words_{5}".format(dm, vector_size, window, epoch, hs, dbow_words)
        if (name_model in existing_model) and write:
            f.write("Model already exists \n")
            print("Model already saved in local")
        
        else:
            
            if write:
                date_begin = datetime.now()
                f.write("Training began at : {0} \n".format(date_begin))

            model = gensim.models.doc2vec.Doc2Vec(dm=dm, vector_size=vector_size, \
                                                  window=window, epoch=epoch, hs=hs,
                                                  dbow_words=dbow_words)
            model.build_vocab(train_corpus)

            if write:
                date_end_vocab = datetime.now()
                f.write("Finished building vocabulary at : {0}, took : {1} \n".format(date_end_vocab, \
                                                                                      date_end_vocab - date_begin))
                date_begin_train = datetime.now()

            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

            if write:
                date_end_train = datetime.now()
                f.write("Finished training model at : {0}, took : {1} \n".format(date_end_train, \
                                                                                 date_end_train - date_begin_train))
            print("Finished training")
            model.save('models_svm/{0}'.format(name_model))
            print("Saved model")
            print("")

            if write:
                date_end = datetime.now()
                f.write("Full process for this model ended at : {0}, took: {1}".format(date_end,
                                                                                date_end - date_begin))
                f.write('\n')

    if write: f.close()
        
        
if __name__ == '__main__':
    
    # Initializing training corpus + parameters
    train_corpus = list(read_corpus(SVM_TRAIN_FILE_DIR))
    dm_val = [0, 1]  # If 1 dm, else dbow
    vector_size_val = [50, 100]  # 100 good enough for us
    window_val = [2, 4, 10, 15]
    epochs_val = [20, 40]  # 10 or 20
    hs_val = [0, 1]  # If 1 hierarchical softmax
    dbow_words_val = [0, 1]  # 1: train word vectors jointly
    params = [dm_val, vector_size_val, window_val, epochs_val, hs_val, dbow_words_val]

    train_save(params, train_corpus)