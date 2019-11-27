# -*- coding: utf-8 -*-
import json
from datetime import datetime
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from helpers.helpers_cv import folder_round_robin
from data_model.formatted_data import X_train, y_train
from data_model.formatted_data import X_dev, y_dev
from part_ii_svm.doc_embeddings import SVMBOW
from ressources.settings import PATH_NEG_TAG, PATH_POS_TAG, FREQ_CUTOFF



param_grid = {'svm__kernel': ['linear', 'rbf', 'poly'],
              'svm__C': [0.1, 1, 10],
              'svm__degree': [2, 3, 5],
              'svm__gamma': ['scale', 'auto', 1, 0.1, 0.01, 10],
            }


# Main script running for this experiment
if __name__ == '__main__':
  # Preparing .txt file
  curr_date = str(datetime.now())
  y, mo, d, h, mi, s = tuple(curr_date[:10].split('-') + curr_date[11:19].split(':'))
  file_name_no_ext = "part_ii_svm/experiments/SVM_BOW_results_{0}_{1}_{2}_{3}_{4}_{5}".format(y, mo, d, h, mi, s)

  # Creating .txt file with logs
  f= open("{0}.txt".format(file_name_no_ext),"w+")
  f.write("Results for CV - SVM BOW - {0} \n".format(curr_date))
  f.write("==========")
  f.write("\n")
  all_results = {"true_pred": y_dev, "models_res": {}}

  # Finding best hyperparameters for each set of parameters (type_feat, tf_idf, norm)
  # Fine tuning on X_train and y_train (2nd to 9th folder of RR folders)
  # Evaluating on X_test and y_test (1st folder of RR folders)


  for t in ['unigram', 'bigram', 'joint']:
      for tf_idf in [0, 1]:
        for norm in [True, False]:
            model_name = "feat_type_{0}_tf_idf_{1}_norm_{2}".format(t, tf_idf, norm)
            f.write("Model fine tuned for {0} : feat => {1}, tf_idf => {2}, norm => {3} \n".format(model_name, t, tf_idf, norm))
            date_begin = datetime.now()

            # GridSearch CV process
            f.write("Process for GridSearchCV began at: {0} \n".format(date_begin))
            pipe_log = Pipeline([('svmbow', SVMBOW(t=t, freq_cutoff=FREQ_CUTOFF, tf_idf=tf_idf, norm=norm)), 
                                 ('svm', SVC(gamma='scale'))])
            log_grid = GridSearchCV(pipe_log, param_grid=param_grid, scoring="accuracy",
                                    verbose=3, cv=5, n_jobs=-1)

            fitted = log_grid.fit(X_train, y_train)
            date_end_cv = datetime.now()
            f.write("Gridsearch CV ended at : {0}, took : {1} \n".format(date_end_cv, date_end_cv - date_begin))
            best_params, best_score = log_grid.best_params_, log_grid.best_score_
            f.write("Best params: {0} \n".format(best_params))
            f.write("Best score: {0} \n".format(best_score))

            # Evaluating on dev set
            f.write("Evaluating on dev set \n")
            date_begin_dev = datetime.now()
            f.write("Process for dev set began at: {0} \n".format(date_begin_dev))
            clf = Pipeline([('svmbow', SVMBOW(t=t, freq_cutoff=FREQ_CUTOFF, tf_idf=tf_idf, norm=norm)), 
                            ('svm', SVC(gamma='scale'))])
            clf.set_params(**best_params)
            clf.fit(X_train, y_train)
            y_predicted = clf.predict(X_dev)
            date_end_dev = datetime.now()
            f.write("Fitting and predicting dev ended at : {0}, took : {1} \n".format(date_end_dev, date_end_dev - date_begin_dev))
            f.write("Accuracy for this model is : {0} \n".format(np.mean(np.array(y_predicted)==np.array(y_dev))))
            f.write("Full process took: {0} \n".format(date_end_dev - date_begin))
            f.write("==========")
            f.write('\n')

            # Storing results
            all_results["models_res"][model_name] = y_predicted
  
  f.close()

  # Saving results as json
  with open("{0}.json".format(file_name_no_ext),"w") as fp:
      json.dump(all_results, fp, default=str)
            
            
