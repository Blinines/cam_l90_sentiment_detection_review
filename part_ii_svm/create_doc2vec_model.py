# -*- coding: utf-8 -*-
import smart_open
import gensim
import itertools
import collections
from gensim.models.doc2vec import Doc2Vec
from os import listdir
from datetime import datetime
from ressources.settings import PATH_PROJECT

svm_train_folder_dir = ['aclImdb/test/neg/', 'aclImdb/test/pos/', 
                        'aclImdb/train/neg/', 'aclImdb/train/pos/', 'aclImdb/train/unsup/']
svm_train_file_dir = []
for folder in svm_train_folder_dir:
    svm_train_file_dir += [PATH_PROJECT + folder + file_name \
                                for file_name in listdir(PATH_PROJECT + folder)]


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


# Initializing training corpus + parameters
train_corpus = list(read_corpus(svm_train_file_dir))
dm_val = [0, 1]
vector_size_val = [50, 100]
window_val = [2, 4]
epochs_val = [20, 40]
hs_val = [0, 1]
params = [dm_val, vector_size_val, window_val, epochs_val, hs_val]


train_save = False
if train_save: # Training and saving all possible models
    f= open("train_doc2vec_model_{0}.txt".format(str(datetime.now())[:10]),"w+")
    for param in itertools.product(*params):
        dm, vector_size, window, epoch, hs = param[0], param[1], param[2], param[3], param[4]
        f.write("dm: {0}, vector_size: {1}, window: {2}, epoch: {3}, hs: {4} \n" \
                    .format(dm, vector_size, window, epoch, hs))
        print("Model : dm: {0}, vector_size: {1}, window: {2}, epoch: {3}, hs: {4}".format(dm, vector_size, window, epoch, hs))
        date_begin = datetime.now()
        f.write("Training began at : {0} \n".format(date_begin))

        model = gensim.models.doc2vec.Doc2Vec(dm=dm, vector_size=vector_size, \
                                              window=window, epoch=epoch, hs=hs)
        model.build_vocab(train_corpus)
        date_end_vocab = datetime.now()
        f.write("Finished building vocabulary at : {0}, took : {1} \n".format(date_end_vocab, \
                                                                              date_end_vocab - date_begin))
        date_begin_train = datetime.now()
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        date_end_train = datetime.now()
        f.write("Finished training model at : {0}, took : {1} \n".format(date_end_train, \
                                                                         date_end_train - date_begin_train))
        print("Finished training")
        name_model = "models_svm/dm_{0}_vector_size_{1}window_{2}epoch_{3}hs_{4}".format(dm, vector_size, window, epoch, hs)
        model.save(name_model)
        print("Saved model")
        print("")
        date_end = datetime.now()
        f.write("Full process for this model ended at : {0}, took: {1}".format(date_end,
                                                                               date_end - date_begin))
        f.write('\n')

    f.close()


assess_model = False
if assess_model:
    f= open("assess_doc2vec_models_{0}.txt".format(str(datetime.now())[:10]),"w+")
    models = listdir("model/")
    for model_name in models:
        model = Doc2Vec.load("models_svm/{0}".format(model_name))
        ranks = []
        second_ranks = []
        for doc_id in range(len(train_corpus)):
            inferred_vector = model.infer_vector(train_corpus[doc_id].words)
            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)
        counter = collections.Counter(ranks)
        f.write("{0} \n".format(model))
        f.write("Counter: {0} \n".format(counter))
        f.write("\n")
    f.close()
        
        