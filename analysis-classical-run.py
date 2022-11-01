

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
                            "--dataset", "manifesto-8",
                            "--languages", "en", "de", "es", "fr", "tr", "ru", "ko",
                            "--language_anchor", "en", "--language_train", "en",  # in multiling scenario --language_train is not used
                            "--augmentation_nmt", "many2many",  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
                            "--sample_interval", "300",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "classical_ml", "--model", "logistic",  # SVM, logistic
                            "--vectorizer", "embeddings-multi",  # "tfidf", "embeddings-en", "embeddings-multi"
                            "--nmt_model", "m2m_100_1.2B",  #"m2m_100_1.2B", "m2m_100_418M"
                            "--hyperparam_study_date", "20221026"])


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
  df_cl = pd.read_csv("./data-clean/df_manifesto_all.csv")
  df_train = pd.read_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}_embed_tfidf.csv")
  df_test = pd.read_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}_embed_tfidf.csv")
else:
  raise Exception(f"Dataset name not found: {DATASET}")

## special preparation of manifesto simple dataset - chose 8 or 57 labels
if DATASET == "manifesto-8":
  df_cl["label_text"] = df_cl["label_domain_text"]
  df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]
  df_train["label_text"] = df_train["label_domain_text"]
  df_train["label"] = pd.factorize(df_train["label_text"], sort=True)[0]
  df_test["label_text"] = df_test["label_domain_text"]
  df_test["label"] = pd.factorize(df_test["label_text"], sort=True)[0]
else:
    raise Exception(f"Dataset not defined: {DATASET}")

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
from helpers import select_data_for_scenario_hp_search, select_data_for_scenario_final_test, data_augmentation, sample_for_scenario, choose_preprocessed_text





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
    hp_study_dic = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{HYPERPARAM_STUDY_DATE}.pkl")
    hp_study_dic = next(iter(hp_study_dic.values()))  # unnest dic
elif EXECUTION_TERMINAL == False:
    #hp_study_dic = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_many2many_embeddings-en_{n_sample_string}samp_{HYPERPARAM_STUDY_DATE}.pkl")
    hp_study_dic = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{HYPERPARAM_STUDY_DATE}.pkl")
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

    print("Number of test examples after sampling: ", len(df_test_scenario))
    print("Number of training examples after sampling: ", len(df_train_scenario_samp))

    if n_sample == 0:  # only one inference necessary on same test set in case of zero-shot
      metric_step = {'accuracy_balanced': 0, 'accuracy_not_b': 0, 'f1_macro': 0, 'f1_micro': 0}
      f1_macro_lst.append(0)
      f1_micro_lst.append(0)
      accuracy_balanced_lst.append(0)
      experiment_details_dic_lang.update({"n_train_total": len(df_train_scenario_samp), f"metrics_seed_{random_seed_sample}_lang_{lang}": metric_step})
      break


    ### data augmentation on sample for multiling models + translation scenarios
    # general function - common with hp-search script
    df_train_scenario_samp_augment = data_augmentation(df_train_scenario_samp=df_train_scenario_samp, df_train=df_train, lang=lang, augmentation=AUGMENTATION, vectorizer=VECTORIZER, language_train=LANGUAGE_TRAIN, language_anchor=LANGUAGE_ANCHOR)

    print("Number of training examples after (potential) augmentation: ", len(df_train_scenario_samp_augment))

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




"""
##### single lang experiments
### classical_ml
## no-NMT-single, sentence-embed-multi
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-single', 
'f1_macro_mean': 0.3534645474830353, 'f1_micro_mean': 0.4572575832546144, 'accuracy_balanced_mean': 0.35691265644956, 'f1_macro_mean_std': 0.012277780071643304, 'f1_micro_mean_std': 0.014746932856258629, 'accuracy_balanced_mean_std': 0.010978804500949638}}

## anchor 'en', tfidf, one2anchor
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'tfidf', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.2118354405320051, 'f1_micro_mean': 0.3179280368464983, 'accuracy_balanced_mean': 0.2163801405620471, 'f1_macro_mean_std': 0.009229978878317351, 'f1_micro_mean_std': 0.011363037915812729, 'accuracy_balanced_mean_std': 0.008312693863799151}}

## anchor 'en', sentence-embeddings-en, one2anchor
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-en', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.33984091289212187, 'f1_micro_mean': 0.43704185004817836, 'accuracy_balanced_mean': 0.3455634379787608, 'f1_macro_mean_std': 0.012872255743693268, 'f1_micro_mean_std': 0.011860833391862844, 'accuracy_balanced_mean_std': 0.01098504906461055}}

## anchor 'en', sentence-embeddings-multi, trained on one2anchor (EN-anchor+anchor2test-lang), tested on test-lang 
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.34753720700622903, 'f1_micro_mean': 0.4462708519608439, 'accuracy_balanced_mean': 0.3514446040496088, 'f1_macro_mean_std': 0.01092396332476347, 'f1_micro_mean_std': 0.0140087481234386, 'accuracy_balanced_mean_std': 0.010391447194617548}}
# ! seems embed-multi seems actually better when no additional embeddings for translations. probably adds unnecessary noise from lower quality texts through nmt  # simple no-NMT-single seems better

## sentence-embeddings-multi, trained on one2many, tested on test-lang 
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2many', 
'f1_macro_mean': 0.334976068703692, 'f1_micro_mean': 0.43052156503444294, 'accuracy_balanced_mean': 0.3373919447972773, 'f1_macro_mean_std': 0.015057241526948756, 'f1_micro_mean_std': 0.010832965715384535, 'accuracy_balanced_mean_std': 0.011477658939804574}}


### standard_dl
## no-NMT-single, minilm-multi, 30 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-single', 
'f1_macro_mean': 0.32465740077048505, 'f1_micro_mean': 0.4350816756328619, 'accuracy_balanced_mean': 0.33964981863190635, 'f1_macro_mean_std': 0.017021589438980484, 'f1_micro_mean_std': 0.008622679567886487, 'accuracy_balanced_mean_std': 0.015569686061854549}}

## one2anchor, minilm-en, 30 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/MiniLM-L12-H384-uncased', 'vectorizer': 'transformer-mono', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.3526785087677711, 'f1_micro_mean': 0.4449542193991269, 'accuracy_balanced_mean': 0.3624410341419119, 'f1_macro_mean_std': 0.0182065168918998, 'f1_micro_mean_std': 0.02047364185883419, 'accuracy_balanced_mean_std': 0.01922773673919911}}

## one2anchor, minilm-multi, 30 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.3310419058216454, 'f1_micro_mean': 0.4276589497148626, 'accuracy_balanced_mean': 0.3431239237844909, 'f1_macro_mean_std': 0.012778547620174932, 'f1_micro_mean_std': 0.011620658524093516, 'accuracy_balanced_mean_std': 0.00975351471176684}}

## one2many, minilm-multi, 15 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2many', 
'f1_macro_mean': 0.34504896603553187, 'f1_micro_mean': 0.45934509552937114, 'accuracy_balanced_mean': 0.3520232568809181, 'f1_macro_mean_std': 0.016290885919853306, 'f1_micro_mean_std': 0.011691613391657663, 'accuracy_balanced_mean_std': 0.01616228799569507}}



#### many lang experiments
### classical_ml

# no-nmt-many, tfidf
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'tfidf', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-many',
'f1_macro_mean': 0.19921951661655699, 'f1_micro_mean': 0.3009558921594391, 'accuracy_balanced_mean': 0.2019695591798122, 'f1_macro_mean_std': 0.009561234583870556, 'f1_micro_mean_std': 0.00615240006513217, 'accuracy_balanced_mean_std': 0.008381838229723854}}
# ! problematic because no custom stopwords & lang-specific feature engineering

# no-nmt-many, embeddings-en (embeddings-multi separately per lang as proxy)
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-en', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-many', 
'f1_macro_mean': 0.3685606391115852, 'f1_micro_mean': 0.47229330204043096, 'accuracy_balanced_mean': 0.36848250108324637, 'f1_macro_mean_std': 0.0132849060127712, 'f1_micro_mean_std': 0.01544597252506312, 'accuracy_balanced_mean_std': 0.014226180844586689}}

# no-nmt-many, embeddings-multi (not separately)
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-many', 
'f1_macro_mean': 0.40123910112737377, 'f1_micro_mean': 0.4962471131639723, 'accuracy_balanced_mean': 0.39881525293622316, 'f1_macro_mean_std': 0.012050144455204859, 'f1_micro_mean_std': 0.007794457274826777, 'accuracy_balanced_mean_std': 0.010892572269832757}}

# many2anchor, tfidf
# !!! performance with 500 samp is better than 2000 samp => there must be an issue with my code ... (or hps are really bad for larger sample)
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'tfidf', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.14096287931460702, 'f1_micro_mean': 0.29330254041570436, 'accuracy_balanced_mean': 0.16227912164844183, 'f1_macro_mean_std': 0.010291036560354988, 'f1_micro_mean_std': 0.001732101616628151, 'accuracy_balanced_mean_std': 0.005258498354857241}}
# ! not sure why this is worse than no-nmt-many. bug in code? or too much translation noise? should be better because more originally different texts. df_train_samp seems correct. maybe because test set more diverse and larger with NMT noise? Or unsuitable hp!
# with train shuffle
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'tfidf', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.14072795148138845, 'f1_micro_mean': 0.29272517321016167, 'accuracy_balanced_mean': 0.16202710551940958, 'f1_macro_mean_std': 0.010305313432689367, 'f1_micro_mean_std': 0.0017321016166281789, 'accuracy_balanced_mean_std': 0.005258498354857241}}

# many2anchor, embeddings-en
# !!! performance with 500 samp is better than 2000 samp => there must be an issue with my code ... (or hps are really bad for larger sample)
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-en', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.35697876826617514, 'f1_micro_mean': 0.4532332563510393, 'accuracy_balanced_mean': 0.3564410059193819, 'f1_macro_mean_std': 0.008993920383241288, 'f1_micro_mean_std': 0.00808314087759815, 'accuracy_balanced_mean_std': 0.008827801936200363}}
# ! not sure why this is worse than no-nmt-many. bug in code? or too much translation noise? should be better because more originally different texts. df_train_samp seems correct. maybe because test set more diverse and larger with NMT noise? Or unsuitable hp!
# with train shuffle
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-en', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.36208952684838036, 'f1_micro_mean': 0.4636258660508083, 'accuracy_balanced_mean': 0.3636840115696522, 'f1_macro_mean_std': 0.010150913399416511, 'f1_micro_mean_std': 0.006351039260969971, 'accuracy_balanced_mean_std': 0.012020711360166303}}

# many2anchor, embeddings-multi
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor',
'f1_macro_mean': 0.35997759792774364, 'f1_micro_mean': 0.4416859122401848, 'accuracy_balanced_mean': 0.35838894874988086, 'f1_macro_mean_std': 0.008749341694553575, 'f1_micro_mean_std': 0.0017321016166281789, 'accuracy_balanced_mean_std': 0.009085730632123923}}
# ?! translating to anchor and mixing seems to hurt performance quite a bit. Despite mixing original texts with trans-anchor. maybe embeddings cannot represent additional info in single vectors properly? Or unsuitable hps!

# many2many, embeddings-multi
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2many', 
'f1_macro_mean': 0.30696868425830426, 'f1_micro_mean': 0.3767321016166282, 'accuracy_balanced_mean': 0.3076885088264402, 'f1_macro_mean_std': 0.006938317325874105, 'f1_micro_mean_std': 0.0025981524249422683, 'accuracy_balanced_mean_std': 0.007123764810302247}}
# more mixing hurts even more (or less suitable hps)


### standard_dl
## no-NMT-many, minilm-multi, 10 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-many', 
'f1_macro_mean': 0.37265943750685937, 'f1_micro_mean': 0.49855658198614317, 'accuracy_balanced_mean': 0.38138113840610777, 'f1_macro_mean_std': 0.029512500372632644, 'f1_micro_mean_std': 0.025115473441108538, 'accuracy_balanced_mean_std': 0.019593473356427776}}

## many2anchor, minilm-en, 10 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/MiniLM-L12-H384-uncased', 'vectorizer': 'transformer-mono', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.12292358678515314, 'f1_micro_mean': 0.3443995381062356, 'accuracy_balanced_mean': 0.1865493321634295, 'f1_macro_mean_std': 0.06726829055535061, 'f1_micro_mean_std': 0.05802540415704388, 'accuracy_balanced_mean_std': 0.061549332163429504}}
# 30 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/MiniLM-L12-H384-uncased', 'vectorizer': 'transformer-mono', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.29671129778832206, 'f1_micro_mean': 0.398094688221709, 'accuracy_balanced_mean': 0.3096610023639281, 'f1_macro_mean_std': 0.00959969473724312, 'f1_micro_mean_std': 0.012990762124711314, 'accuracy_balanced_mean_std': 0.005635760287067082}}

## many2anchor, minilm-multi, 10 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.40696748699064167, 'f1_micro_mean': 0.5046189376443417, 'accuracy_balanced_mean': 0.4081168572038839, 'f1_macro_mean_std': 0.0013554627823026688, 'f1_micro_mean_std': 0.006351039260969971, 'accuracy_balanced_mean_std': 0.00019480069060628935}}

## many2many, minilm-multi, 4 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2many', 
'f1_macro_mean': 0.4060399798702805, 'f1_micro_mean': 0.4942263279445728, 'accuracy_balanced_mean': 0.41114805502934704, 'f1_macro_mean_std': 0.01256610814450368, 'f1_micro_mean_std': 0.026558891454965372, 'accuracy_balanced_mean_std': 0.01592137067509322}}



"""




