# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import json
from part_i_naive_bayes.naive_bayes import NaiveBayes
from part_i_naive_bayes.analyze_results import analyze_results
from helpers.helpers_cv import RoundRobinCV
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG, FREQ_CUTOFF_UNIGRAM, FREQ_CUTOFF_BIGRAM
from ressources.settings import TYPE_NB, SMOOTHING_NB, FEAT_TYPE, FREQ_CUTOFF

if __name__ == '__main__':
    curr_date = str(datetime.now())
    y, mo, d, h, mi, s = tuple(curr_date[:10].split('-') + curr_date[11:19].split(':'))
    file_name_no_ext = "results/NB_results_{0}_{1}_{2}_{3}_{4}_{5}".format(y, mo, d, h, mi, s)

    # Creating .txt file with logs
    f= open("{0}.txt".format(file_name_no_ext),"w+")
    all_results = {t: {} for t in TYPE_NB}
    for t in TYPE_NB:
        for smoothing in  SMOOTHING_NB:
            all_results[t][smoothing] = {}
            for feat_type in FEAT_TYPE:
                f.write("Type NB: {0}, smoothing: {1}, feature type: {2} \n".format(t, smoothing, feat_type))
                date_begin = datetime.now()
                f.write("Process began at: {0} \n".format(date_begin))

                # CV
                CLF = NaiveBayes(t=t, smoothing=smoothing, 
                                 freq_cutoff=FREQ_CUTOFF, feat_type=feat_type)
                RR = RoundRobinCV(clf=CLF, path_neg=PATH_NEG_TAG, path_pos=PATH_POS_TAG, mod=10)
                results = RR.cross_validate()

                # Storing results
                all_results[t][smoothing][feat_type] = results
                scores = []
                for index in results.keys():
                    score = results[index]["score"]
                    f.write("{0} folder as test data: accuracy is {1} \n".format(index, score))
                    scores.append(score)

                mean_score = np.mean(np.array(scores))
                all_results[t][smoothing][feat_type]["avg_score"] = mean_score
                f.write("Averaged accuracy is {0} \n".format(mean_score))

                date_end = datetime.now()
                f.write("Process ended at: {0} \n".format(date_end))
                f.write("Process took: {0} \n".format(date_end - date_begin))
                f.write('\n')
    f.close()
    
    # Saving results as json
    with open("{0}.json".format(file_name_no_ext),"w") as fp:
        json.dump(all_results, fp, default=str)
    
    # Printing results directly
    analyze_results(json_path="{0}.json".format(file_name_no_ext))

