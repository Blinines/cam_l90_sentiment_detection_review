Sentiment detection of reviews
-----------------------

Replication experiment code of [Bo Pang et al.](https://www.aclweb.org/anthology/W02-1011.pdf), who considered the problem of document classification by overall sentiment.

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

* Make sure you have a correct environment created for this project. You might want to use a virtual environment.  Python 3 preferred. In any case, activate your environment.
 
* Install the requirements using `pip install -r requirements.txt`.

* Install the setup file using `python setup.py install`

* NB : when you make change to files which are used as modules in other files, you will likely need to re run the `python setup.py install` command.

Usage
-----------------------

This README.md is for now under construction. Below are the details regarding the first part of the practical, i.e. specifically focusing on Naive Bayes.

* The main script to run all the experiments is in the `main` folder, by using `python main/main_nb.py`on the terminal command.
    * 3 settings can be changed in the `ressources/settings.py` file, by default they are as follow : 
        * `TYPE_NB = ['unigram', 'bigram', 'joint']` : Type of n-gram to take into account. Only those three can be calculated.
        * `SMOOTHING_NB = [0, 1]`. Smoothing applied.
        * `FEAT_TYPE` = ['freq', 'pres']. How to represent the documents for the testing phase. 
    * Launching the script will compute Round Robin cross validation for all possible models for the settings, and save two new files : one .txt file and one .json file. The .txt file contains readable sentence of results per fold, whereas the .json file will contain raw data.
    * By default launching the script will also analyze the results that will be printed on the terminal command : info both on accuracy and p-value between models will be displayed.

Describing project architecture
----------------------------------

* [data](./data)
    * [NEG](./data/NEG) : positive text reviews
    * [POS](./data/POS) : negative text reviews
* [data-tagged](./data-tagged) 
    * [NEG](./data-tagged/NEG) : positive tagged (already tokenised) reviews
    * [POS](./data-tagged/POS) : negative tagged (already tokenised) reviews
* [helpers](./helpers)
    * [helpers_bow](./helpers/helpers_bow.py) : helpers for building bow
    * [helpers_cv](./helpers/helpers_cv.py) : helpers for cross validation
    * [helpers_gen](./helpers/helpers_gen.py) : general helpers (mostly for statistical tests)
    * [helpers_nb](./helpers/helpers_nb.py) : helpers specific to NB
* [main](./main)
    * [main_nb](./main/main_nb.py) : main script to run for Part I - NB only
* [part_i_naive_bayes](./part_i_naive_bayes)
    * [analyze_results](./part_i_naive_bayes/analyze_results.py) : accuracy info per model and sign test results across models
    * [feat_count](./part_i_naive_bayes/feat_count.py) : feature count before and after frequency cutoff 
    * [naive_bayes](./part_i_naive_bayes/naive_bayes.py) : Naive Bayes class implementation, with methods fit and predict
* [part_ii_svm](./part_ii_svm)
    * [create_doc2vec_model](./part_ii_svm/create_doc2vec_model.py) : training different Doc2Vec models and saving them (locally) in order to work with them faster afterwards
* [private](./private)
    * [private](./private/private.py) : non shared file with access to the data
* [ressources](./ressources)
    * [settings](./ressources/settings.py) : global settings for running algorithms
* [results](./results) : storing various results from experiments
* [.gitignore](./.gitignore) : files to be ignores by Git when committing 
* [main_svm](./main_svm.py)
* [pipeline_svm](./pipeline_svm.py)
* [README.md](./README.md)
* [requirements.txt](./requirements.txt#)
* [setup](./setup.py)
* [svm_bow](./svm_bow.py)
* [svm](./svm.py)


