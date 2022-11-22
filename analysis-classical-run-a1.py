

import sys
# does not work on snellius
"""if sys.stdin.isatty():
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (sys.stdin.isatty(): ", sys.stdin.isatty(), ")")"""
if len(sys.argv) > 1:
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (len(sys.argv): ", len(sys.argv), ")")


# ## Load packages
import transformers
import datasets
import torch
import optuna

import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
import random
import os
import tqdm
from collections import OrderedDict
from datetime import date
import time
import joblib
import ast

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import svm, naive_bayes, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer



## set global seed for reproducibility and against seed hacking
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

print(os.getcwd())
if (EXECUTION_TERMINAL==False) and ("multilingual-repo" not in os.getcwd()):
    os.chdir("./multilingual-repo")
print(os.getcwd())



# ## Main parameters

### argparse for command line execution
import argparse
# https://realpython.com/command-line-interfaces-python-argparse/
# https://docs.python.org/3/library/argparse.html

# Create the parser
parser = argparse.ArgumentParser(description='Do final run with best hyperparameters (on different languages, datasets, algorithms)')

## Add the arguments
# arguments only for test script
parser.add_argument('-cvf', '--n_cross_val_final', type=int, default=3,
                    help='For how many different random samples should the algorithm be tested at a given sample size?')
parser.add_argument('-zeroshot', '--zeroshot', action='store_true',
                    help='Start training run with a zero-shot run')

# arguments for both hyperparam and test script
parser.add_argument('-lang', '--languages', type=str, nargs='+',
                    help='List of languages to iterate over. e.g. "en", "de", "es", "fr", "ko", "tr", "ru" ')
parser.add_argument('-anchor', '--language_anchor', type=str,
                    help='Anchor language to translate all texts to if using anchor. Default is "en"')
parser.add_argument('-language_train', '--language_train', type=str,
                    help='What language should the training set be in?. Default is "en"')
parser.add_argument('-augment', '--augmentation_nmt', type=str,
                    help='Whether and how to augment the data with machine translation (MT).')

parser.add_argument('-ds', '--dataset', type=str,
                    help='Name of dataset. Can be one of: "manifesto-8" ')
parser.add_argument('-samp', '--sample_interval', type=int, nargs='+',
                    help='Interval of sample sizes to test.')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "classical_ml"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model or "SVM" or "logistic". Must fit to "method" argument.')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings-en" or "embeddings-multi"')
parser.add_argument('-hp_date', '--hyperparam_study_date', type=str,
                    help='Date string to specifiy which hyperparameter run should be selected. e.g. "20220304"')
parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')



### choose arguments depending on execution in terminal or in script for testing
if EXECUTION_TERMINAL == True:
  print("Arguments passed via the terminal:")
  # Execute the parse_args() method
  args = parser.parse_args()
  # To show the results of the given option to screen.
  print("")
  for key, value in parser.parse_args()._get_kwargs():
      #if value is not None:
          print(value, "  ", key)

elif EXECUTION_TERMINAL == False:
  # parse args if not in terminal, but in script
  args = parser.parse_args(["--n_cross_val_final", "2",  #--zeroshot
                            "--dataset", "pimpo_samp_a1",
                            "--languages", 'sv', 'no', 'da', 'fi', 'nl', 'es', 'de', 'en', 'fr',  #"en", "de", "es", "fr", "tr", "ru", "ko",
                            "--language_anchor", "en", "--language_train", "en",  # in multiling scenario --language_train is not used
                            "--augmentation_nmt", "many2many",  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
                            "--sample_interval", "100",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "classical_ml", "--model", "logistic",  # SVM, logistic
                            "--vectorizer", "embeddings-multi",  # "tfidf", "embeddings-en", "embeddings-multi"
                            "--nmt_model", "m2m_100_418M",  #"m2m_100_1.2B", "m2m_100_418M"
                            "--hyperparam_study_date", "20221111"])


### args only for test runs
CROSS_VALIDATION_REPETITIONS_FINAL = args.n_cross_val_final
ZEROSHOT = args.zeroshot

### args for both hyperparameter tuning and test runs
# choose dataset
DATASET = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
LANGUAGES = args.languages
LANGUAGES = LANGUAGES
LANGUAGE_TRAIN = args.language_train
LANGUAGE_ANCHOR = args.language_anchor
AUGMENTATION = args.augmentation_nmt

N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set
VECTORIZER = args.vectorizer

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "classical_ml"
MODEL_NAME = args.model  # "SVM"

HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"
NMT_MODEL = args.nmt_model





# ## Load data
if "manifesto-8" in DATASET:
  df_cl = pd.read_csv("./data-clean/df_manifesto_all.zip")
  df_train = pd.read_csv(f"./data-clean/df_{DATASET}_samp_train_trans_{NMT_MODEL}_embed_tfidf.zip")
  df_test = pd.read_csv(f"./data-clean/df_{DATASET}_samp_test_trans_{NMT_MODEL}_embed_tfidf.zip")
if "pimpo_samp_a1" in DATASET:
  df_cl = pd.read_csv("./data-clean/df_pimpo_all.zip")
  df_train = pd.read_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}_embed_tfidf.zip")
  df_test = pd.read_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}_embed_tfidf.zip")
  # only doing analysis on immigration
  df_cl = df_cl[df_cl.label_text.isin(['no_topic', 'immigration_sceptical', 'immigration_supportive', 'immigration_neutral'])]  # 'integration_neutral', 'integration_sceptical', 'integration_supportive'
  df_train = df_train[df_train.label_text.isin(['no_topic', 'immigration_sceptical', 'immigration_supportive', 'immigration_neutral'])]  # 'integration_neutral', 'integration_sceptical', 'integration_supportive'
  df_test = df_test[df_test.label_text.isin(['no_topic', 'immigration_sceptical', 'immigration_supportive', 'immigration_neutral'])]  # 'integration_neutral', 'integration_sceptical', 'integration_supportive'
else:
  raise Exception(f"Dataset name not found: {DATASET}")

## special preparation of manifesto simple dataset - chose 8 or 57 labels
if "manifesto-8" in DATASET:
  df_cl["label_text"] = df_cl["label_domain_text"]
  df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]
  df_train["label_text"] = df_train["label_domain_text"]
  df_train["label"] = pd.factorize(df_train["label_text"], sort=True)[0]
  df_test["label_text"] = df_test["label_domain_text"]
  df_test["label"] = pd.factorize(df_test["label_text"], sort=True)[0]
elif "pimpo_samp_a1" in DATASET:
  df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]
  df_train["label"] = pd.factorize(df_train["label_text"], sort=True)[0]
  df_test["label"] = pd.factorize(df_test["label_text"], sort=True)[0]
else:
    Exception(f"Dataset not defined: {DATASET}")

print(DATASET)




## reduce max sample size interval list to fit to max df_train length
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample <= len(df_train)]
if len(df_train) < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len(df_train)]
  if len(n_sample_dev_filt) > 1:
    if n_sample_dev_filt[-1] == n_sample_dev_filt[-2]:  # if last two sample sizes are duplicates, delete the last one
      n_sample_dev_filt = n_sample_dev_filt[:-1]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)

"""
# tests for code above
N_SAMPLE_DEV = [1000]
len_df_train = 500
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample <= len_df_train]
if len_df_train < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len_df_train]
  if len(n_sample_dev_filt) > 1:
    if n_sample_dev_filt[-1] == n_sample_dev_filt[-2]:  # if last two sample sizes are duplicates, delete the last one
      n_sample_dev_filt = n_sample_dev_filt[:-1]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)"""



LABEL_TEXT_ALPHABETICAL = np.sort(df_cl.label_text.unique())
TRAINING_DIRECTORY = f"results/{DATASET}"


## data checks
print(DATASET, "\n")
# verify that numeric label is in alphabetical order of label_text
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").label.tolist()  # label num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text
assert all(labels_num_via_numeric == labels_num_via_text)


# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())

import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)

from helpers import compute_metrics_classical_ml, clean_memory
## functions for scenario data selection and augmentation
from helpers import select_data_for_scenario_final_test, data_augmentation, choose_preprocessed_text





# ## Final test with best hyperparameters

## hyperparameters for final tests
# selective load one decent set of hps for testing
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results/manifesto-8/optuna_study_SVM_tfidf_01000samp_20221006.pkl")

# select best hp based on hp-search
n_sample = N_SAMPLE_DEV[0]
n_sample_string = N_SAMPLE_DEV[0]
#n_sample_string = 300
while len(str(n_sample_string)) <= 4:
    n_sample_string = "0" + str(n_sample_string)

if EXECUTION_TERMINAL == True:
    hp_study_dic = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{DATASET}_{NMT_MODEL}_{HYPERPARAM_STUDY_DATE}.pkl")
    hp_study_dic = next(iter(hp_study_dic.values()))  # unnest dic
elif EXECUTION_TERMINAL == False:
    #hp_study_dic = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_many2many_embeddings-en_{n_sample_string}samp_{HYPERPARAM_STUDY_DATE}.pkl")
    hp_study_dic = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{DATASET}_{NMT_MODEL}_{HYPERPARAM_STUDY_DATE}.pkl")
    hp_study_dic = next(iter(hp_study_dic.values()))  # unnest dic


# this implementation means that the terminal argument with languages is effectively ignored. added assert test to ensure equality - depends on scenario whether that's an issue
if ZEROSHOT == False:
    if "language" in hp_study_dic.keys():  # for scenario where only one hp-search,
        HYPER_PARAMS_LST = [hp_study_dic["optuna_study"].best_trial.user_attrs["hyperparameters_all"]]
        HYPER_PARAMS_LST = HYPER_PARAMS_LST * len(hp_study_dic["language"])
        # ! important that first lang in this list is not used somewhere downstream
        LANG_LST = hp_study_dic["language"]  #["run-only-once"]  # different string to make sure that this is not actually used downstream - should not be used for these scenarios
        assert LANG_LST == LANGUAGES, "The languages from the hp_study_dic are not the same as the languages passed as an argument"
    else:  # for scenario where multiple/different hp-searchers per lang
        HYPER_PARAMS_LST = [value_scenario_dic['optuna_study'].best_trial.user_attrs["hyperparameters_all"] for key_lang, value_scenario_dic in hp_study_dic.items()]
        LANG_LST = [key_lang for key_lang, value_scenario_dic in hp_study_dic.items()]
        assert LANG_LST == LANGUAGES, "The languages from the hp_study_dic are not the same as the languages passed as an argument"
else:
    raise Exception("zero-shot classification not implemented")





#### K example intervals loop
# will automatically only run once if only 1 element in HYPER_PARAMS_LST for runs with one good hp-value
experiment_details_dic = {}
n_language = 0
for lang, hyperparams in tqdm.tqdm(zip(LANG_LST, HYPER_PARAMS_LST), desc="Iterations for different languages and hps", leave=True):
  n_language += 1
  np.random.seed(SEED_GLOBAL)
  t_start = time.time()  # log how long training of model takes

  ### select correct language scenario for train sampling and testing
  df_train_scenario, df_test_scenario = select_data_for_scenario_final_test(df_train=df_train, df_test=df_test, lang=lang, augmentation=AUGMENTATION, vectorizer=VECTORIZER, language_train=LANGUAGE_TRAIN, language_anchor=LANGUAGE_ANCHOR)


  # prepare loop
  experiment_details_dic_lang = {"method": METHOD, "language_source": lang, "language_anchor": LANGUAGE_ANCHOR, "n_sample": n_sample, "model": MODEL_NAME, "vectorizer": VECTORIZER, "augmentation": AUGMENTATION, "hyperparams": hyperparams}  # "trainer_args": train_args, "hypotheses": HYPOTHESIS_TYPE, "dataset_stats": dataset_stats_dic
  f1_macro_lst = []
  f1_micro_lst = []
  accuracy_balanced_lst = []
  # randomness stability loop. Objective: calculate F1 across N samples to test for influence of different (random) samples
  np.random.seed(SEED_GLOBAL)
  if n_language == 1:
      model_dic = {}  # storing trained classifiers to avoid re-training when not necessary
  for random_seed_sample in tqdm.tqdm(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_FINAL), desc="iterations for std", leave=True):
    print("Random seed: ", random_seed_sample)

    ## take sample in accordance with scenario
    if n_sample == 999_999:  # all data, no sampling
      df_train_scenario_samp = df_train_scenario.copy(deep=True)
    # same sample size per language for multiple language data scenarios
    elif AUGMENTATION in ["no-nmt-many", "many2anchor", "many2many"]:
      df_train_scenario_samp = df_train_scenario.groupby(by="language_iso").apply(lambda x: x.sample(n=min(n_sample, len(x)), random_state=random_seed_sample).copy(deep=True))
    # one sample size for single language data scenario
    else:
      df_train_scenario_samp = df_train_scenario.sample(n=min(n_sample, len(df_train_scenario)), random_state=random_seed_sample).copy(deep=True)
    # faulty function for sampling
    #df_train_scenario_samp = sample_for_scenario_final(df_train_scenario=df_train_scenario, n_sample=n_sample, augmentation=AUGMENTATION, vectorizer=VECTORIZER, seed=SEED_GLOBAL, lang=lang, dataset=DATASET)

    print("Number of training examples after sampling: ", len(df_train_scenario_samp))
    print("Label distribution for df_train_scenario_samp: ", df_train_scenario_samp.label_text.value_counts())
    print("Language distribution for df_train_scenario_samp: ", df_train_scenario_samp.language_iso.value_counts())
    print("Number of test examples (should be constant): ", len(df_test_scenario))
    print("Label distribution for df_test_scenario: ", df_test_scenario.label_text.value_counts())
    print("\n")


    if n_sample == 0:  # only one inference necessary on same test set in case of zero-shot
      metric_step = {'accuracy_balanced': 0, 'accuracy_not_b': 0, 'f1_macro': 0, 'f1_micro': 0}
      f1_macro_lst.append(0)
      f1_micro_lst.append(0)
      accuracy_balanced_lst.append(0)
      experiment_details_dic_lang.update({"n_train_total": len(df_train_scenario_samp), f"metrics_seed_{random_seed_sample}_lang_{lang}": metric_step})
      break


    ### data augmentation on sample for multiling models + translation scenarios
    # general function - common with hp-search script
    df_train_scenario_samp_augment = data_augmentation(df_train_scenario_samp=df_train_scenario_samp, df_train=df_train, lang=lang, augmentation=AUGMENTATION, vectorizer=VECTORIZER, language_train=LANGUAGE_TRAIN, language_anchor=LANGUAGE_ANCHOR, dataset=DATASET)

    print("Number of training examples after (potential) augmentation: ", len(df_train_scenario_samp_augment))

    print("\nCounts for checking augmentation issues: ")
    print("\nCount for df_train_scenario_samp_augment.language_iso: ", df_train_scenario_samp_augment.language_iso.value_counts())
    print("Count for df_train_scenario_samp_augment.language_iso_trans: ", df_train_scenario_samp_augment.language_iso_trans.value_counts())

    print("\nCount for df_test_scenario.language_iso: ", df_test_scenario.language_iso.value_counts())
    print("Count for df_test_scenario.language_iso_trans: ", df_test_scenario.language_iso_trans.value_counts())

    ### text pre-processing
    # separate hyperparams for vectorizer and classifier.
    hyperparams_vectorizer = {key: value for key, value in hyperparams.items() if key in ["ngram_range", "max_df", "min_df", "analyzer"]}
    hyperparams_clf = {key: value for key, value in hyperparams.items() if key not in ["ngram_range", "max_df", "min_df", "analyzer"]}
    vectorizer_sklearn = TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", use_idf=True, smooth_idf=True, **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02, token_pattern="(?u)\b\w\w+\b"

    # choose correct pre-processed text column here. possible vectorizers: "tfidf", "embeddings-en", "embeddings-multi"
    X_train, X_test = choose_preprocessed_text(df_train_scenario_samp_augment=df_train_scenario_samp_augment, df_test_scenario=df_test_scenario, augmentation=AUGMENTATION, vectorizer=VECTORIZER, vectorizer_sklearn=vectorizer_sklearn, language_train=LANGUAGE_TRAIN, language_anchor=LANGUAGE_ANCHOR, method=METHOD)


    y_train = df_train_scenario_samp_augment.label
    y_test = df_test_scenario.label


    # training on train set sample
    # train 1 (*n_seeds) or 7 (*n_seeds) classifiers, depending on scenario
    # these scenarios needs re-training
    if (n_language == 1) or ((AUGMENTATION in ["no-nmt-many", "many2many"]) and (VECTORIZER in ["tfidf", "embeddings-en"])) or ((AUGMENTATION in ["one2anchor"]) and (VECTORIZER in ["embeddings-multi", "tfidf"])) or ((AUGMENTATION in ["many2anchor"]) and (VECTORIZER in ["tfidf"])):  # tfidf in here, because classifier expects always same feature input length as it has been trained on. this varies for tfidf across languages
        if MODEL_NAME == "SVM":
            clf = svm.SVC(**hyperparams_clf)
        elif MODEL_NAME == "logistic":
            clf = linear_model.LogisticRegression(**hyperparams_clf)
        clf.fit(X_train, y_train)
        print("Training new classifier")
        if n_language == 1:
            model_dic.update({random_seed_sample: clf})
    # otherwise, re-use previous classifier
    else:
        print("! Skipping training of new classifier, because can reuse previous one !")
        clf = model_dic[random_seed_sample]



    # prediction on test set
    label_gold = y_test
    label_pred = clf.predict(X_test)

    ### metrics
    metric_step = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df_cl.label_text.unique()))

    experiment_details_dic_lang.update({f"metrics_seed_{random_seed_sample}_lang_{lang}": metric_step})
    f1_macro_lst.append(metric_step["eval_f1_macro"])
    f1_micro_lst.append(metric_step["eval_f1_micro"])
    accuracy_balanced_lst.append(metric_step["eval_accuracy_balanced"])

    print(f"evaluation done for lang:  {lang}")
    np.random.seed(SEED_GLOBAL)

    if (n_sample == 0) or (n_sample == 999_999):  # only one inference necessary on same test set in case of zero-shot or full dataset
      break


  # timer
  t_end = time.time()
  t_total = round(t_end - t_start, 2)
  t_total = t_total / CROSS_VALIDATION_REPETITIONS_FINAL  # average of all random seed runs

  ## calculate aggregate metrics across random runs
  f1_macro_mean = np.mean(f1_macro_lst)
  f1_micro_mean = np.mean(f1_micro_lst)
  accuracy_balanced_mean = np.mean(accuracy_balanced_lst)
  f1_macro_std = np.std(f1_macro_lst)
  f1_micro_std = np.std(f1_micro_lst)
  accuracy_balanced_std = np.std(accuracy_balanced_lst)
  # add aggregate metrics to overall experiment dict
  metrics_mean = {"f1_macro_mean": f1_macro_mean, "f1_micro_mean": f1_micro_mean, "accuracy_balanced_mean": accuracy_balanced_mean, "f1_macro_std": f1_macro_std, "f1_micro_std": f1_micro_std, "accuracy_balanced_std": accuracy_balanced_std}
  experiment_details_dic_lang.update({"metrics_mean": metrics_mean, "dataset": DATASET, "n_train_total": len(df_train_scenario_samp), "n_classes": len(df_cl.label_text.unique()), "train_eval_time_per_model": t_total})


  # update of overall experiment dic
  experiment_details_dic.update({f"experiment_sample_{n_sample_string}_{METHOD}_{MODEL_NAME}_{lang}": experiment_details_dic_lang})


  ## stop loop for multiple language case - no separate iterations per language necessary
  #if "many2anchor" in AUGMENTATION:
  #  break
  #if ("multi" in VECTORIZER) and (any(augmentation in AUGMENTATION for augmentation in ["no-nmt-many", "many2many"])):  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
  #  break



### summary dictionary across all languages
experiment_summary_dic = {"experiment_summary":
     {"dataset": DATASET, "model_name": MODEL_NAME, "vectorizer": VECTORIZER, "augmentation": AUGMENTATION, "lang_anchor": LANGUAGE_ANCHOR, "lang_train": LANGUAGE_TRAIN, "lang_all": LANGUAGES, "method": METHOD, "sample_size": N_SAMPLE_DEV, "nmt_model": NMT_MODEL}
}

# calculate averages across all languages
f1_macro_lst_mean = []
f1_micro_lst_mean = []
accuracy_balanced_lst_mean = []
f1_macro_lst_mean_std = []
f1_micro_lst_mean_std = []
accuracy_balanced_lst_mean_std = []
for experiment_key in experiment_details_dic:
    f1_macro_lst_mean.append(experiment_details_dic[experiment_key]['metrics_mean']['f1_macro_mean'])
    f1_micro_lst_mean.append(experiment_details_dic[experiment_key]['metrics_mean']['f1_micro_mean'])
    accuracy_balanced_lst_mean.append(experiment_details_dic[experiment_key]['metrics_mean']['accuracy_balanced_mean'])
    f1_macro_lst_mean_std.append(experiment_details_dic[experiment_key]['metrics_mean']['f1_macro_std'])
    f1_micro_lst_mean_std.append(experiment_details_dic[experiment_key]['metrics_mean']['f1_micro_std'])
    accuracy_balanced_lst_mean_std.append(experiment_details_dic[experiment_key]['metrics_mean']['accuracy_balanced_std'])
    #print(f"{experiment_key}: f1_macro: {experiment_details_dic[experiment_key]['metrics_mean']['f1_macro_mean']} , f1_micro: {experiment_details_dic[experiment_key]['metrics_mean']['f1_micro_mean']} , accuracy_balanced: {experiment_details_dic[experiment_key]['metrics_mean']['accuracy_balanced_mean']}")
f1_macro_lst_cross_lang_mean = np.mean(f1_macro_lst_mean)
f1_micro_lst_cross_lang_mean = np.mean(f1_micro_lst_mean)
accuracy_balanced_lst_cross_lang_mean = np.mean(accuracy_balanced_lst_mean)
f1_macro_lst_mean_std_mean = np.mean(f1_macro_lst_mean_std)
f1_micro_lst_mean_std_mean = np.mean(f1_micro_lst_mean_std)
accuracy_balanced_lst_mean_std_mean = np.mean(accuracy_balanced_lst_mean_std)
# cross language std
f1_macro_lst_mean_cross_lang_std = np.std(f1_macro_lst_mean)
f1_micro_lst_mean_cross_lang_std = np.std(f1_micro_lst_mean)
accuracy_balanced_lst_mean_cross_lang_std = np.std(accuracy_balanced_lst_mean)

experiment_summary_dic["experiment_summary"].update({"f1_macro_mean": f1_macro_lst_cross_lang_mean, "f1_micro_mean": f1_micro_lst_cross_lang_mean, "accuracy_balanced_mean": accuracy_balanced_lst_cross_lang_mean,
                               "f1_macro_mean_cross_lang_std": f1_macro_lst_mean_cross_lang_std, "f1_micro_mean_cross_lang_std": f1_micro_lst_mean_cross_lang_std, "accuracy_balanced_mean_cross_lang_std": accuracy_balanced_lst_mean_cross_lang_std,
                               "f1_macro_mean_std_mean": f1_macro_lst_mean_std_mean, "f1_micro_mean_std_mean": f1_micro_lst_mean_std_mean, "accuracy_balanced_mean_std_mean": accuracy_balanced_lst_mean_std_mean})

print(experiment_summary_dic)


### save full experiment dic
# merge individual languages experiments with summary dic
experiment_details_dic_summary = {**experiment_details_dic, **experiment_summary_dic}


if EXECUTION_TERMINAL == True:
  joblib.dump(experiment_details_dic_summary, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{DATASET}_{NMT_MODEL}_{HYPERPARAM_STUDY_DATE}.pkl")
elif EXECUTION_TERMINAL == False:
  joblib.dump(experiment_details_dic_summary, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{DATASET}_{NMT_MODEL}_{HYPERPARAM_STUDY_DATE}_t.pkl")



### double checking for issues
print("\n\nChecking against issues: ")
print("\nNumber of test examples: ", len(df_test_scenario))
print("Number of training examples before augmentation: ", len(df_train_scenario_samp))
print("Number of training examples after (potential) augmentation: ", len(df_train_scenario_samp_augment))

print("\nCounts for checking augmentation issues: ")
print("df train language counts: ")
print("Count for df_train_scenario_samp_augment.language_iso: ", df_train_scenario_samp_augment.language_iso.value_counts())
print("Count for df_train_scenario_samp_augment.language_iso_trans: ", df_train_scenario_samp_augment.language_iso_trans.value_counts())
print("df test language counts: ")
print("Count for df_test_scenario.language_iso: ", df_test_scenario.language_iso.value_counts())
print("Count for df_test_scenario.language_iso_trans: ", df_test_scenario.language_iso_trans.value_counts())
print("\n")

for experiment_key in experiment_details_dic:
    if experiment_key != "experiment_summary":
        print("source lang: ", experiment_details_dic[experiment_key]["language_source"])
        print("anchor lang: ", experiment_details_dic[experiment_key]["language_anchor"])
        for metric_seed_keys in [key for key in experiment_details_dic[experiment_key].keys() if "metrics_seed" in str(key)]:
            print(metric_seed_keys, ": ", experiment_details_dic[experiment_key][metric_seed_keys]["eval_f1_macro"])




print("\n\nRun done.")


