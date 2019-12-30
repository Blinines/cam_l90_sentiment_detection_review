Sentiment detection of reviews
-----------------------

Replication experiment code of [Bo Pang et al.](https://www.aclweb.org/anthology/W02-1011.pdf), who considered the problem of document classification by overall sentiment.
Additionnally to this approach we also implement a SVM approach with Doc2Vec document embeddings.

Installation
----------------------

#### Settings 
In case you accessed the code via the VM, everything is already installed hence you can skip to the next section

* Clone this repo to your computer by using `git clone https://github.com/Blinines/cam_l90_sentiment_detection_review.git` in the folder you want the repo to be in.
* Get into the folder using `cd cam_l90_sentiment_detection_review`.
* Inside the folder `private`, create a `private.py` file. This file should contain several parameters that you need to set up before using the scripts.
    * Information to find the datasets : PATH_PROJECT. Inside this folder there should be direct access to the data tagged and untagged.
* Ensure this `private.py` file is int  the .gitignore so Git will ignore this when you modify the code.


#### Install the requirements

* Make sure you have a correct environment created for this project. You might want to use a virtual environment.  Python 3 preferred. In any case, activate your environment. For Linux the command is `source venv/bin/activate`, for Windows it is `venv\Scripts\activate`, if the name of your virtual environment is `venv`.
 
* Install the requirements using `pip install -r requirements.txt`.

* Install the setup file using `python setup.py install`

* NB : when you make change to files which are used as modules in other files, you will likely need to re run the `python setup.py install` command.

Usage
-----------------------

This usage sections follows the two different parts of the practical. The first part was specifically focusing on the Naive Bayes approaches, whereas the second one also included SVM approaches using bag-of-words representations and Doc2Vec document embeddings.


* For the Naive Bayes approach only : the main script to run all the experiments is in the `main` folder, by using `python main/main_nb.py`on the terminal command. (supposing root location of the project)
    * 3 settings can be changed in the `ressources/settings.py` file, by default they are as follow : 
        * `TYPE_NB = ['unigram', 'bigram', 'joint']` : Type of n-gram to take into account. Only those three can be calculated.
        * `SMOOTHING_NB = [0, 1]`. Smoothing applied.
        * `FEAT_TYPE = ['freq']`. How to represent the documents for the testing phase. 
    * Launching the script will compute Round Robin cross validation for all possible models for the settings, and save two new files : one .txt file and one .json file. The .txt file contains readable sentence of results per fold, whereas the .json file will contain raw data. One latest example is available in the VM.
    * By default launching the script will also analyze the results that will be printed on the terminal command : info both on accuracy and p-value between models will be displayed. p-value was computed using the sign test.

* For all the final experiments : the main script to run all the experiments is in the `main` folder, by using `python main/main_all.py`on the terminal command. (supposing root location of the project)
    * Each best model from previous experiments is trained and results are predicted. Models include : the three Naive Bayes approaches with unigrams, bigrams and joint unigrams and bigrams respectively, the SVM approaches with unigrams, bigrams and joint unigrams and bigrams bag-of-words representation, the SVM approach with the Dov2Vec document embedding.
    * Launching the script will also analyze the results that will be printed on the terminal command : info both on accuracy and p-value between models will be displayed. p-value was computed using the Monte Carlo permutation test.
    * Results are then stored in the `results` folder. The name of the file begins by `all_results` and resembles the date and time the experiment began afterwards.

Describing project architecture
----------------------------------

* [data](./data)  : given dataset, .txt format
    * [NEG](./data/NEG) : positive text reviews
    * [POS](./data/POS) : negative text reviews
* [data_model](./data_model)
    * [formatted_data](./data_model/formatted_data.py) : from Round Robin folders, defining training, development and test sets.
* [data-tagged](./data-tagged)  : given dataset, .tag format
    * [NEG](./data-tagged/NEG) : positive tagged (already tokenised) reviews
    * [POS](./data-tagged/POS) : negative tagged (already tokenised) reviews
* [helpers](./helpers)
    * [helpers_bow](./helpers/helpers_bow.py) : helpers for building bow (building features and bow)
    * [helpers_cv](./helpers/helpers_cv.py) : helpers for cross validation (separating folders)
    * [helpers_gen](./helpers/helpers_gen.py) : general helpers (mostly for statistical tests, sign test and permutation test)
    * [helpers_nb](./helpers/helpers_nb.py) : helpers specific to NB (frequency bow, Bayes' rule)
* [main](./main)
    * [main_all](./main/main_all.py) : main script to run for Part II - NB and SVM approaches
    * [main_nb](./main/main_nb.py) : main script to run for Part I - NB only
* [part_i_naive_bayes](./part_i_naive_bayes)
    * [analyze_results](./part_i_naive_bayes/analyze_results.py) : accuracy info per model and sign test results across models
    * [feat_count](./part_i_naive_bayes/feat_count.py) : feature count before and after frequency cutoff 
    * [naive_bayes](./part_i_naive_bayes/naive_bayes.py) : Naive Bayes class implementation, with methods fit and predict
* [part_ii_svm](./part_ii_svm)
    * [experiments](./part_ii_svm/experiments) : containing both .json and .txt files regarding experiments made on SVM models. (cross-validation, training Doc2Vec models)
    * [models_svm](./part_ii_svm/models_svm) : saved trained models for Doc2Vec embeddings. Space consuming hence specific to each local project.
    * [analyze_cv_all](./part_ii_svm/analyze_cv_all.py) : analyzing results from models using SVM approach
    * [create_doc2vec_model](./part_ii_svm/create_doc2vec_model.py) : training different Doc2Vec models and saving them (locally) in order to work with them faster afterwards
    * [cv_bow](./part_ii_svm/cv_bow.py) : cross-validation to get best hyperparametersfor BOW-SVM approaches
    * [cv_doc2vec_hyper_param](./part_ii_svm/cv_doc2vec_hyper_param) : cross-validation to get best hyperparametersfor Doc2Vec-SVM approaches. Emphasis on two important hyperparameters, i.e. dm vs dbow architecture, and jointly training words and documents or not.
    * [docs_embeddings](./part_ii_svm/docs_embeddings) : SVM classes implementation (BOW and Doc2Vec), with methods fit and transform. Later used in the code within a pipeline.
    * [visualisation.ipynb](./part_ii_svm/visualisation.ipynb) : analysing embedding space. _Requires the installation of module jupyter to be displayed in VSC_
* [private](./private)
    * [private](./private/private.py) : non shared file with access to the project folder
* [ressources](./ressources)
    * [settings](./ressources/settings.py) : global settings for running algorithms
* [results](./results) : storing various results from experiments
* [.gitignore](./.gitignore) 
* [bibliography.bib](./bibliography.bib) : bibliography used for datasets and implementation. Bibtex format.
* [README.md](./README.md)
* [requirements.txt](./requirements.txt#)
* [setup](./setup.py)


