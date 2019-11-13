# -*- coding: utf-8 -*-
from datetime import datetime
import json
from part_i_naive_bayes.naive_bayes import NaiveBayes
from helpers.helpers_cv import RoundRobinCV
from helpers.helpers_gen import get_accuracy
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG, FREQ_CUTOFF_UNIGRAM, FREQ_CUTOFF_BIGRAM

if __name__ == '__main__':
    f= open("results_{0}.txt".format(str(datetime.now())[:10]),"w+")
    all_results = {t: {0: None, 1:None} for t in ['unigram', 'bigram', 'joint']}
    for t in ['unigram', 'bigram', 'joint']:
        for smoothing in  [0, 1]:
            feat_type = 'pres'
            f.write("Feature type: {0}".format(feat_type))
            f.write("Type NB: {0}, smoothing: {1} \n".format(t, smoothing))
            date_begin = datetime.now()
            f.write("Process began at: {0} \n".format(date_begin))
            CLF = NaiveBayes(t=t, smoothing=smoothing, 
                             freq_cutoff={1: FREQ_CUTOFF_UNIGRAM, 2: FREQ_CUTOFF_BIGRAM},
                             feat_type=feat_type)
            RR = RoundRobinCV(clf=CLF, path_neg=PATH_NEG_TAG, path_pos=PATH_POS_TAG, mod=10)
            results = RR.cross_validate()
            all_results[t][smoothing] = results
            for index in results.keys():
                f.write("{0} folder as test data: accuracy is {1} \n".format(index, get_accuracy(results[index])))
            date_end = datetime.now()
            f.write("Process ended at: {0} \n".format(date_end))
            f.write("Process took: {0} \n".format(date_end - date_begin))
            f.write('\n')

    f.close()

    with open("results_{0}.json".format(str(datetime.now())[:10]),"w") as fp:
        json.dump(all_results, fp)