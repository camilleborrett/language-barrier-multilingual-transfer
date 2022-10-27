#!/usr/bin/env python
# coding: utf-8

import sys
if sys.stdin.isatty():
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (sys.stdin.isatty(): ", sys.stdin.isatty(), ")")


# ## Load packages
import transformers
import datasets
import torch
import optuna

import pandas as pd
import numpy as np
import re
import math
import random
import os
import tqdm
from collections import OrderedDict
import joblib
from datetime import date
from datetime import datetime
import ast

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import svm, naive_bayes, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

import spacy


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
parser = argparse.ArgumentParser(description='Run hyperparameter tuning with different languages, algorithms, datasets')

## Add the arguments
# arguments for hyperparameter search
parser.add_argument('-t', '--n_trials', type=int,
                    help='How many optuna trials should be run?')
parser.add_argument('-ts', '--n_trials_sampling', type=int,
                    help='After how many trials should optuna start sampling?')
parser.add_argument('-tp', '--n_trials_pruning', type=int,
                    help='After how many trials should optuna start pruning?')
parser.add_argument('-cvh', '--n_cross_val_hyperparam', type=int, default=2,
                    help='How many times should optuna cross validate in a single trial?')
#parser.add_argument('-context', '--context', action='store_true',
#                    help='Take surrounding context sentences into account.')

# arguments for both hyperparam and test script
parser.add_argument('-lang', '--languages', type=str, nargs='+',
                    help='List of languages to iterate over. e.g. "en", "de", "es", "fr", "ko", "tr", "ru" ')
parser.add_argument('-anchor', '--language_anchor', type=str,
                    help='Anchor language to translate all texts to if not many2many. Default is "en"')
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
                    help='How to vectorize text. Options: "tfidf" or "embeddings"')
#parser.add_argument('-tqdm', '--disable_tqdm', action='store_true',
#                    help='Adding the flag enables tqdm for progress tracking')
#parser.add_argument('-carbon', '--carbon_tracking', action='store_true',
#                    help='Adding the flag enables carbon tracking via CodeCarbon')  # not used, as CodeCarbon caused bugs https://github.com/mlco2/codecarbon/issues/305

parser.add_argument('-hp_date', '--hyperparam_study_date', type=str,
                    help='Date string to specifiy which hyperparameter run should be selected. e.g. "20220304"')




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
  args = parser.parse_args(["--n_trials", "10", "--n_trials_sampling", "7", "--n_trials_pruning", "7", "--n_cross_val_hyperparam", "2",  #"--context",
                            "--dataset", "manifesto-8",
                            "--languages", "en", "de", "es", "fr", "tr", "ru", "ko",
                            "--language_anchor", "en", "--language_train", "en",  # in multiling scenario --language_train needs to be list of lang (?)
                            "--augmentation_nmt", "many2many",  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
                            "--sample_interval", "50",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "classical_ml", "--model", "logistic",  # SVM, logistic
                            "--vectorizer", "embeddings-multi",  # "tfidf", "embeddings-en", "embeddings-multi"
                            "--hyperparam_study_date", "20221026"])


### args only for hyperparameter tuning
N_TRIALS = args.n_trials
N_STARTUP_TRIALS_SAMPLING = args.n_trials_sampling
N_STARTUP_TRIALS_PRUNING = args.n_trials_pruning
CROSS_VALIDATION_REPETITIONS_HYPERPARAM = args.n_cross_val_hyperparam
#CONTEXT = False  #args.context   # ! do not use context, because some languages have too little data and makes train-test split problematic

### args for both hyperparameter tuning and test runs
# choose dataset
DATASET_NAME = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
LANGUAGES = args.languages
LANGUAGE_TRAIN = args.language_train
LANGUAGE_ANCHOR = args.language_anchor
AUGMENTATION = args.augmentation_nmt

N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set
VECTORIZER = args.vectorizer

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "classical_ml"
MODEL_NAME = args.model  # "SVM"

HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"
#DISABLE_TQDM = args.disable_tqdm
# = args.carbon_tracking




# ## Load data
if "manifesto-8" in DATASET_NAME:
  df_cl = pd.read_csv("./data-clean/df_manifesto_all.csv")
  df_train = pd.read_csv("./data-clean/df_manifesto_train_trans_embed_tfidf.csv")
  df_test = pd.read_csv("./data-clean/df_manifesto_test_trans_embed_tfidf.csv")
else:
  raise Exception(f"Dataset name not found: {DATASET_NAME}")

## special preparation of manifesto simple dataset - chose 8 or 57 labels
if DATASET_NAME == "manifesto-8":
  df_cl["label_text"] = df_cl["label_domain_text"]
  df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]
  df_train["label_text"] = df_train["label_domain_text"]
  df_train["label"] = pd.factorize(df_train["label_text"], sort=True)[0]
  df_test["label_text"] = df_test["label_domain_text"]
  df_test["label"] = pd.factorize(df_test["label_text"], sort=True)[0]
else:
    Exception(f"Dataset not defined: {DATASET_NAME}")

print(DATASET_NAME)


## fill na for non translated texts to facilitate downstream analysis
#df_train["language_iso_trans"] = [language_iso if pd.isna(language_iso_trans) else language_iso_trans for language_iso, language_iso_trans in zip(df_train.language_iso, df_train.language_iso_trans)]
#df_train["text_original_trans"] = [text_original if pd.isna(text_original_trans) else text_original_trans for text_original, text_original_trans in zip(df_train.text_original, df_train.text_original_trans)]
#df_test["language_iso_trans"] = [language_iso if pd.isna(language_iso_trans) else language_iso_trans for language_iso, language_iso_trans in zip(df_test.language_iso, df_test.language_iso_trans)]
#df_test["text_original_trans"] = [text_original if pd.isna(text_original_trans) else text_original_trans for text_original, text_original_trans in zip(df_test.text_original, df_test.text_original_trans)]



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
TRAINING_DIRECTORY = f"results/{DATASET_NAME}"


## data checks
print(DATASET_NAME, "\n")
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

from helpers import compute_metrics_classical_ml, clean_memory  # data_preparation



# !! do not need this anymore without NLI
### load suitable hypotheses_hyperparameters and text formatting function
"""def format_text(df=None, text_format=None, embeddings=embeddings):
    if text_format == 'template_not_nli':
        df["text_prepared"] = df.text
    elif text_format == 'template_quote':
        df["text_prepared"] = 'The quote: "' + df.text + '" - end of the quote.'
    elif text_format == 'template_complex':
        df["text_prepared"] = df.text
    else:
        raise Exception(f'Hypothesis template not found for: {text_format}')
    return df.copy(deep=True)"""



#from hypothesis_hyperparams import hypothesis_hyperparams
### load the hypothesis hyperparameters for the respective dataset. For classical_ml this only determines the input text format - sentence with surrounding sentences, or not
#hypothesis_hyperparams_dic, format_text = hypothesis_hyperparams(dataset_name=DATASET_NAME, df_cl=df_cl)

# check which template fits to standard_dl/classical_ml or NLI
#print("")
#print([template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template])
#print([template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" not in template])

# in case -context flag is passed, but dataset actually does not have context
# ! do not use context, because some languages have too little data and makes train-test split problematic
#classical_templates = [template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template]
#if "context" not in classical_templates:
#  CONTEXT = False




##### prepare texts for classical ML TF-IDF
## embeddings from transformers are already added in data frame upstream

## lemmatize text
# ! lemmatization would require different model for each new language. Seems fair to assume that this does not exist in multilingual situation
"""if VECTORIZER == "tfidf":
    nlp = spacy.load("en_core_web_md")
    np.random.seed(SEED_GLOBAL)
    def lemmatize(text_lst):
        texts_lemma = []
        for doc in nlp.pipe(text_lst, n_process=4):  # disable=["tok2vec", "ner"] "tagger", "attribute_ruler", "parser",
            doc_lemmas = " ".join([token.lemma_ for token in doc])
            texts_lemma.append(doc_lemmas)
        return texts_lemma

    ## lemmatize depending on dataset
    if "text_original" in df_cl.columns:
        df_train["text_original"] = lemmatize(df_train.text_original)
        #df_test["text_original"] = lemmatize(df_test.text_original)
    else:
        df_train["text"] = lemmatize(df_train.text)
        #df_test["text"] = lemmatize(df_test.text)"""





### select correct language for training sampling and validation
def select_data_for_scenario_hp_search(df_train=None, df_dev=None, lang=None):
    ## augmentations for certain scenarios in code outside of this function. needs to be afterwards because need to first sample, then augment

    ## Two different language scenarios
    if "no-nmt-single" in AUGMENTATION:
        df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
        df_dev_lang = df_dev.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
        if "multi" not in VECTORIZER:
            raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")
    elif "one2anchor" in AUGMENTATION:
        # for test set - for non-multi, test on translated text, for multi algos test on original lang text
        if "multi" not in VECTORIZER:
            df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
            df_dev_lang = df_dev.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
        elif "multi" in VECTORIZER:
            # could add augmentation for this scenario downstream (with anchor)
            df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
            df_dev_lang = df_dev.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
    elif "one2many" in AUGMENTATION:
        # augmenting this with other translations further down after sampling
        df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
        # for test set - for multi algos test on original lang text
        if "multi" in VECTORIZER:
            df_dev_lang = df_dev.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
        elif "multi" not in VECTORIZER:
            raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")

    ## many2X scenarios
    elif "no-nmt-many" in AUGMENTATION:
        # separate analysis per lang if not multi
        if "multi" not in VECTORIZER:
            df_train_lang = df_train.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
            df_dev_lang = df_dev.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
        # multilingual models can analyse all original texts here
        elif "multi" in VECTORIZER:
            df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
            df_dev_lang = df_dev.query("language_iso == language_iso_trans").copy(deep=True)
    elif "many2anchor" in AUGMENTATION:
        if "multi" not in VECTORIZER:
            df_train_lang = df_train.query("language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
            df_dev_lang = df_dev.query("language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
        # multilingual models can analyse all original texts here. can augment below with anchor lang
        elif "multi" in VECTORIZER:
            df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
            df_dev_lang = df_dev.query("language_iso == language_iso_trans").copy(deep=True)
    elif "many2many" in AUGMENTATION:
        if "multi" not in VECTORIZER:
            df_train_lang = df_train.query("language_iso_trans == @lang").copy(deep=True)
            df_dev_lang = df_dev.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
            #raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")
        # multilingual models can analyse all original texts here. can be augmented below with all other translations
        elif "multi" in VECTORIZER:
            df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
            df_dev_lang = df_dev.query("language_iso == language_iso_trans").copy(deep=True)
    else:
        raise Exception("Issue with AUGMENTATION")

    return df_train_lang, df_dev_lang


### select correct language for train sampling and final test
def select_data_for_scenario_final_test(df_train=None, df_test=None, lang=None):
    ## Two different language scenarios
    if "no-nmt-single" in AUGMENTATION:
        df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
        df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
        if "multi" not in VECTORIZER:
            raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")
    elif "one2anchor" in AUGMENTATION:
        df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
        # for test set - for non-multi, test on translated text, for multi algos test on original lang text
        if "multi" not in VECTORIZER:
            df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
        elif "multi" in VECTORIZER:
            df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
    elif "one2many" in AUGMENTATION:
        # augmenting this with other translations further down after sampling
        df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
        # for test set - for multi algos test on original lang text
        if "multi" in VECTORIZER:
            df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
        elif "multi" not in VECTORIZER:
            raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")

    ## many2X scenarios
    elif "no-nmt-many" in AUGMENTATION:
        # separate analysis per lang if not multi
        if "multi" not in VECTORIZER:
            df_train_lang = df_train.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
            df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
        # multilingual models can analyse all original texts here
        elif "multi" in VECTORIZER:
            df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
            df_test_lang = df_test.query("language_iso == language_iso_trans").copy(deep=True)
    elif "many2anchor" in AUGMENTATION:
        if "multi" not in VECTORIZER:
            df_train_lang = df_train.query("language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
            df_test_lang = df_test.query("language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
        # multilingual models can analyse all original texts here. augmented below with anchor lang
        elif "multi" in VECTORIZER:
            df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
            df_test_lang = df_test.query("language_iso == language_iso_trans").copy(deep=True)
    elif "many2many" in AUGMENTATION:
        # multilingual models can analyse all original texts here. can be augmented below with all other translations
        if "multi" not in VECTORIZER:
            df_train_lang = df_train.query("language_iso_trans == @lang").copy(deep=True)
            df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
        elif "multi" in VECTORIZER:
            df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
            df_test_lang = df_test.query("language_iso == language_iso_trans").copy(deep=True)
        #elif "multi" not in VECTORIZER:
        #    # ! can add not multi scenarios here too
        #    raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")
    else:
        raise Exception("Issue with AUGMENTATION")

    return df_train_lang, df_test_lang


### data augmentation for multiling models + translation scenarios
def data_augmentation(df_train_scenario_samp=None, df_train=None):
    ## single language text scenarios
    sample_sent_id = df_train_scenario_samp.sentence_id.unique()
    if AUGMENTATION == "no-nmt-single":
        df_train_scenario_samp_augment = df_train_scenario_samp.copy(deep=True)
    elif AUGMENTATION == "one2anchor":
        if "multi" not in VECTORIZER:
            df_train_scenario_samp_augment = df_train_scenario_samp.copy(deep=True)
        elif "multi" in VECTORIZER:
            # augment by combining texts from train language with texts from train translated to target language
            df_train_augment = df_train[(((df_train.language_iso == LANGUAGE_TRAIN) & (df_train.language_iso_trans == LANGUAGE_TRAIN)) | ((df_train.language_iso == LANGUAGE_TRAIN) & (df_train.language_iso_trans == lang)))].copy(deep=True)
            df_train_scenario_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)
        else:
            raise Exception("Issue with VECTORIZER")
    elif AUGMENTATION == "one2many":
        if "multi" in VECTORIZER:
            df_train_augment = df_train.query("language_iso == @LANGUAGE_TRAIN").copy(deep=True)  # can use all translations and original text (for single train lang) here
            df_train_scenario_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
        else:
            raise Exception("AUGMENTATION == 'X2many' only works for multilingual vectorization")
    ## multiple language text scenarios
    elif AUGMENTATION == "no-nmt-many":
        df_train_scenario_samp_augment = df_train_scenario_samp.copy(deep=True)
    elif AUGMENTATION == "many2anchor":
        if "multi" not in VECTORIZER:
            df_train_scenario_samp_augment = df_train_scenario_samp.copy(deep=True)
        elif "multi" in VECTORIZER:
            # already have all original languages in the scenario. augmenting it with translated (to anchor) texts here. e.g. for 7*6=3500 original texts, adding 6*6=3000 texts, all translated to anchor (except anchor texts)
            df_train_augment = df_train[(df_train.language_iso == df_train.language_iso_trans) | (df_train.language_iso_trans == LANGUAGE_ANCHOR)].copy(deep=True)
            df_train_scenario_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
    elif AUGMENTATION == "many2many":
        if "multi" not in VECTORIZER:
            df_train_scenario_samp_augment = df_train_scenario_samp.copy(deep=True)
        elif "multi" in VECTORIZER:
            # already have all original languages in the scenario. augmenting it with all other translated texts
            df_train_augment = df_train.copy(deep=True)
            df_train_scenario_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
        #else:
        #    raise Exception(f"AUGMENTATION == {AUGMENTATION} only works for multilingual vectorization")
    return df_train_scenario_samp_augment




##### hyperparameter tuning

# carbon tracker  https://github.com/mlco2/codecarbon/tree/master
#if CARBON_TRACKING:
#  from codecarbon import OfflineEmissionsTracker
#  tracker = OfflineEmissionsTracker(country_iso_code="NLD",  log_level='warning', measure_power_secs=300,  #output_dir=TRAINING_DIRECTORY
#                                    project_name=f"{DATASET_NAME}-{MODEL_NAME.split('/')[-1]}")
#  tracker.start()


def optuna_objective(trial, lang=None, n_sample=None, df_train=None, df=None):  # hypothesis_hyperparams_dic=None
  clean_memory()
  np.random.seed(SEED_GLOBAL)  # setting seed again for safety. not sure why this needs to be run here at each iteration. it should stay constant once set globally?! explanation could be this https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f

  # for testing
  global df_train_split
  global df_train_scenario
  global df_train_scenario_samp
  global df_train_scenario_samp_augment
  global df_dev_scenario
  global df_dev_scenario_samp
  global df_train_scenario_samp_ids
  global df_dev_scenario_samp_ids

  if VECTORIZER == "tfidf":
      # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
      hyperparams_vectorizer = {
          'ngram_range': trial.suggest_categorical("ngram_range", [(1, 2), (1, 3), (1, 6)]),
          'max_df': trial.suggest_categorical("max_df", [0.95, 0.9, 0.8]),
          'min_df': trial.suggest_categorical("min_df", [0.01, 0.03, 0.05]),
          'analyzer': trial.suggest_categorical("analyzer", ["word", "char_wb"]),  # could be good for languages like Korean where longer sequences of characters without a space seem to represent compound words
      }
      vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", use_idf=True, smooth_idf=True, **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02, token_pattern="(?u)\b\w\w+\b"
  if  "embeddings" in VECTORIZER:
      hyperparams_vectorizer = {}

  # SVM  # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  if MODEL_NAME == "SVM":
      hyperparams_clf = {'kernel': trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                   'C': trial.suggest_float("C", 1, 1000, log=True),
                   "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                   "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                   "coef0": trial.suggest_float("coef0", 1, 100, log=True),  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
                   "degree": trial.suggest_int("degree", 1, 50, log=False),  # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
                   #"decision_function_shape": trial.suggest_categorical("decision_function_shape", ["ovo", "ovr"]),  # "However, one-vs-one (‘ovo’) is always used as multi-class strategy. The parameter is ignored for binary classification."
                   #"tol": trial.suggest_categorical("tol", [1e-3, 1e-4]),
                   "random_state": SEED_GLOBAL,
                    # 10k max_iter had 1-10% performance drop for high n sample (possibly overfitting to majority class)
                    #MAX_ITER_LOW, MAX_ITER_HIGH = 1_000, 7_000  # tried 10k, but led to worse performance on larger, imbalanced dataset (?)
                    "max_iter": trial.suggest_int("num_train_epochs", 1_000, 7_000, log=False, step=1000),  #MAX_ITER,
                   }
  # Logistic Regression # ! disadvantage: several parameters only work with certain other parameters  # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn-linear-model-logisticregression
  elif MODEL_NAME == "logistic":
      hyperparams_clf = {#'penalty': trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"]),
                        'penalty': 'l2',  # works with all solvers
                        'solver': trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
                        'C': trial.suggest_float("C", 1, 1000, log=False),
                        #"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                        #"intercept_scaling": trial.suggest_float("intercept_scaling", 1, 50, log=False),
                        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                        "max_iter": trial.suggest_int("max_iter", 50, 1000, log=False),  # 100 default
                        "multi_class": "auto",  # {‘auto’, ‘ovr’, ‘multinomial’}
                        "warm_start": trial.suggest_categorical("warm_start", [True, False]),
                        #"l1_ratio": None,
                        "n_jobs": -1,
                        "random_state": SEED_GLOBAL,
                        }
  else:
      raise Exception("Method not available: ", MODEL_NAME)
  
  # not choosing a hypothesis template here, but the way of formatting the input text (e.g. with preceding sentence or not). need to keep same object names
  # if statements determine, whether surrounding sentences are added, or not. Disactivated, because best to always try and test context
  #if CONTEXT == True:
  #  text_template_classical_ml = [template for template in list(hypothesis_hyperparams_dic.keys()) if ("not_nli" in template) and ("context" in template)]
  #elif CONTEXT == False:
  #text_template_classical_ml = [template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template]   #and ("context" in template)
  #else:
  #  raise Exception(f"CONTEXT variable is {CONTEXT}. Can only be True/False")

  #if len(text_template_classical_ml) >= 2:  # if there is only one reasonable text format for standard_dl
  #  hypothesis_template = trial.suggest_categorical("hypothesis_template", text_template_classical_ml)
  #else:
  #  hypothesis_template = text_template_classical_ml[0]

  hyperparams_optuna = {**hyperparams_clf, **hyperparams_vectorizer}  # "hypothesis_template": hypothesis_template
  trial.set_user_attr("hyperparameters_all", hyperparams_optuna)
  print("Hyperparameters for this run: ", hyperparams_optuna)


  # cross-validation loop. Objective: determine F1 for specific sample for specific hyperparams, without a test set
  run_info_dic_lst = []
  for step_i, random_seed_cross_val in enumerate(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_HYPERPARAM)):
    # delete function, was only necessary for NLI
    #df_train_lang_samp, df_dev_lang_samp = data_preparation(random_seed=random_seed_cross_val, df_train=df_train, df=df,
                                                  #hypothesis_template=hypothesis_template,
                                                  #hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template],
                                                  #format_text_func=format_text,
    #                                              n_sample=n_sample, method=METHOD, embeddings=embeddings)

    ## train-validation split
    # ~50% split cross-val as recommended by https://arxiv.org/pdf/2109.12742.pdf
    test_size = 0.4
    # test unique splitting with sentence_id
    df_train_split_ids, df_dev_ids = train_test_split(df_train.sentence_id.unique(), test_size=test_size, shuffle=True, random_state=random_seed_cross_val)
    df_train_split = df_train[df_train.sentence_id.isin(df_train_split_ids)]
    df_dev = df_train[df_train.sentence_id.isin(df_dev_ids)]
    #df_train_split, df_dev = train_test_split(df_train, test_size=test_size, shuffle=True, random_state=random_seed_cross_val, stratify=df_train["stratify_by"])  # stratify_by is language_iso + label_subcat_text; could also use language_trans_iso
    print(f"Final train test length after cross-val split: len(df_train_lang_samp) = {len(df_train_split)}, len(df_dev_lang_samp) {len(df_dev)}.")

    # select data for specific language & augmentation scenario. this should also deduplicate (probably not for multiling scenario)
    df_train_scenario, df_dev_scenario = select_data_for_scenario_hp_search(df_train=df_train_split, df_dev=df_dev, lang=lang)

    ## take sample size for scenario (always 1k for this paper)
    #if AUGMENTATION in ["one2many", "many2many"]:
    #    n_sample_augment = n_sample * len(LANGUAGES)
    #else:
    #    n_sample_augment = n_sample
    if n_sample == 999_999:
        df_train_scenario_samp = df_train_scenario.copy(deep=True)
        df_dev_scenario_samp = df_dev_scenario.copy(deep=True)
    elif AUGMENTATION in ["no-nmt-single", "one2anchor", "one2many"]:
        df_train_scenario_samp_ids = df_train_scenario.sentence_id.sample(n=min(int(n_sample * (1-test_size)), len(df_train_scenario)), random_state=random_seed_cross_val).copy(deep=True)
        df_dev_scenario_samp_ids = df_dev_scenario.sentence_id.sample(n=min(int(n_sample * test_size), len(df_dev_scenario)), random_state=random_seed_cross_val).copy(deep=True)
        df_train_scenario_samp = df_train_scenario[df_train_scenario.sentence_id.isin(df_train_scenario_samp_ids)]
        df_dev_scenario_samp = df_dev_scenario[df_dev_scenario.sentence_id.isin(df_dev_scenario_samp_ids)]
    elif AUGMENTATION in ["no-nmt-many", "many2anchor", "many2many"]:
        df_train_scenario_samp_ids = df_train_scenario.groupby(by="language_iso", group_keys=True, as_index=True).apply(lambda x: x.sentence_id.sample(n=min(int(n_sample * (1-test_size)), len(x)), random_state=random_seed_cross_val))
        df_dev_scenario_samp_ids = df_dev_scenario.groupby(by="language_iso", group_keys=True, as_index=True).apply(lambda x: x.sentence_id.sample(n=min(int(n_sample * test_size), len(x)), random_state=random_seed_cross_val))
        if AUGMENTATION in ["no-nmt-many"] and VECTORIZER in ["tfidf", "embeddings-en"]:
            # unelegant - need to extract row with ids from df - when only run on one language, groupby returns df with separate rows for each lang  (only one lang), but need just one series with all ids
            df_train_scenario_samp_ids = df_train_scenario_samp_ids.loc[lang]
            df_dev_scenario_samp_ids = df_dev_scenario_samp_ids.loc[lang]
        elif AUGMENTATION in ["many2many"] and VECTORIZER in ["tfidf", "embeddings-en"]:
            # unelegant - need to extract row with ids from df - when only run on one language, groupby returns df with separate rows for each lang  (only one lang), but need just one series with all ids
            df_dev_scenario_samp_ids = df_dev_scenario_samp_ids.loc[lang]
        df_train_scenario_samp = df_train_scenario[df_train_scenario.sentence_id.isin(df_train_scenario_samp_ids)]
        df_dev_scenario_samp = df_dev_scenario[df_dev_scenario.sentence_id.isin(df_dev_scenario_samp_ids)]

    print("Number of training examples after sampling: ", len(df_train_scenario_samp))
    print("Number of validation examples after sampling: ", len(df_dev_scenario_samp))


    ### data augmentation on sample for multiling model + MT scenarios
    df_train_scenario_samp_augment = data_augmentation(df_train_scenario_samp=df_train_scenario_samp, df_train=df_train_split)
    print("Number of training examples after possible augmentation: ", len(df_train_scenario_samp_augment))


    clean_memory()
    ## ! choose correct pre-processed text column here
    # possible vectorizers: "tfidf", "embeddings-en", "embeddings-multi"
    if VECTORIZER == "tfidf":
        # fit vectorizer on entire dataset - theoretically leads to some leakage on feature distribution in TFIDF (but is very fast, could be done for each test. And seems to be common practice) - OOV is actually relevant disadvantage of classical ML  #https://github.com/vanatteveldt/ecosent/blob/master/src/data-processing/19_svm_gridsearch.py
        vectorizer.fit(pd.concat([df_train_scenario_samp_augment.text_original_trans_tfidf, df_dev_scenario_samp.text_original_trans_tfidf]))
        X_train = vectorizer.transform(df_train_scenario_samp_augment.text_original_trans_tfidf)
        X_test = vectorizer.transform(df_dev_scenario_samp.text_original_trans_tfidf)
    if "embeddings" in VECTORIZER:
        # possible augmentations: "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
        if (AUGMENTATION in ["one2anchor", "many2anchor"]) and (VECTORIZER == "embeddings-en") and (LANGUAGE_ANCHOR == "en"):
            #X_train = np.array([list(lst) for lst in df_train_scenario_samp_augment.text_original_trans_embed_en])
            #X_test = np.array([list(lst) for lst in df_dev_scenario_samp.text_original_trans_embed_en])
            X_train = np.array([ast.literal_eval(lst) for lst in df_train_scenario_samp_augment.text_original_trans_embed_en.astype('object')])
            X_test = np.array([ast.literal_eval(lst) for lst in df_dev_scenario_samp.text_original_trans_embed_en.astype('object')])
        elif (AUGMENTATION in ["no-nmt-many", "many2many"] and (VECTORIZER == "embeddings-en")):
            # need to use multiling embeddings for this scenario, because no good encoder for each language
            #X_train = np.array([list(lst) for lst in df_train_scenario_samp_augment.text_original_trans_embed_multi])
            #X_test = np.array([list(lst) for lst in df_dev_scenario_samp.text_original_trans_embed_multi])
            X_train = np.array([ast.literal_eval(lst) for lst in df_train_scenario_samp_augment.text_original_trans_embed_multi.astype('object')])
            X_test = np.array([ast.literal_eval(lst) for lst in df_dev_scenario_samp.text_original_trans_embed_multi.astype('object')])
        elif (AUGMENTATION in ["no-nmt-single", "no-nmt-many", "one2anchor", "one2many", "many2anchor", "many2many"]) and (VECTORIZER == "embeddings-multi"):
            # maybe implement taking embeddings-en here if train-lang is en as benchmark  (complicated to implement given all the augmentations above - mixes different languages)
            #X_train = np.array([list(lst) for lst in df_train_scenario_samp_augment.text_original_trans_embed_multi])
            #X_test = np.array([list(lst) for lst in df_dev_scenario_samp.text_original_trans_embed_multi])
            X_train = np.array([ast.literal_eval(lst) for lst in df_train_scenario_samp_augment.text_original_trans_embed_multi.astype('object')])
            X_test = np.array([ast.literal_eval(lst) for lst in df_dev_scenario_samp.text_original_trans_embed_multi.astype('object')])
        else:
            raise Exception(f"No implementation/issue for vectorizer {VECTORIZER} and/or augmentation {AUGMENTATION}")

    y_train = df_train_scenario_samp_augment.label
    y_test = df_dev_scenario_samp.label

    # training on train set sample
    if MODEL_NAME == "SVM":
        clf = svm.SVC(**hyperparams_clf)
    elif MODEL_NAME == "logistic":
        clf = linear_model.LogisticRegression(**hyperparams_clf)
    clf.fit(X_train, y_train)

    # prediction on test set
    label_gold = y_test
    label_pred = clf.predict(X_test)
    
    # metrics
    metric_step = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df.label_text.unique()))

    run_info_dic = {"method": METHOD, "vectorizer": VECTORIZER, "augmentation": AUGMENTATION, "n_sample": n_sample, "model": MODEL_NAME, "results": metric_step, "hyper_params": hyperparams_optuna}
    run_info_dic_lst.append(run_info_dic)
    
    # Report intermediate objective value.
    intermediate_value = (metric_step["eval_f1_macro"] + metric_step["eval_f1_micro"]) / 2
    trial.report(intermediate_value, step_i)
    # Handle trial pruning based on the intermediate value.
    if trial.should_prune() and (CROSS_VALIDATION_REPETITIONS_HYPERPARAM > 1):
      raise optuna.TrialPruned()
    if n_sample == 999_999:  # no cross-validation necessary for full dataset
      break


  ## aggregation over cross-val loop
  f1_macro_crossval_lst = [dic["results"]["eval_f1_macro"] for dic in run_info_dic_lst]
  f1_micro_crossval_lst = [dic["results"]["eval_f1_micro"] for dic in run_info_dic_lst]
  accuracy_balanced_crossval_lst = [dic["results"]["eval_accuracy_balanced"] for dic in run_info_dic_lst]
  metric_details = {
      "F1_macro_mean": np.mean(f1_macro_crossval_lst), "F1_micro_mean": np.mean(f1_micro_crossval_lst), "accuracy_balanced_mean": np.mean(accuracy_balanced_crossval_lst),
      "F1_macro_std": np.std(f1_macro_crossval_lst), "F1_micro_std": np.std(f1_micro_crossval_lst), "accuracy_balanced_std": np.std(accuracy_balanced_crossval_lst)
  }
  trial.set_user_attr("metric_details", metric_details)

  results_lst = [dic["results"] for dic in run_info_dic_lst]
  trial.set_user_attr("results_trainer", results_lst)

  # objective: maximise mean of f1-macro & f1-micro. HP should be good for imbalanced data, but also important/big classes
  metric = (np.mean(f1_macro_crossval_lst) + np.mean(f1_micro_crossval_lst)) / 2
  std = (np.std(f1_macro_crossval_lst) + np.std(f1_micro_crossval_lst)) / 2

  print(f"\nFinal metrics for run: {metric_details}. With hyperparameters: {hyperparams_optuna}\n")

  return metric



#warnings.filterwarnings(action='ignore')
#from requests import HTTPError  # for catching HTTPError, if model download does not work for one trial for some reason
# catch catch following error. unclear if good to catch this. [W 2022-01-12 14:18:30,377] Trial 9 failed because of the following error: HTTPError('504 Server Error: Gateway Time-out for url: https://huggingface.co/api/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')

def run_study(n_sample=None, lang=None):
  np.random.seed(SEED_GLOBAL)

  optuna_pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS_PRUNING, n_warmup_steps=0, interval_steps=1, n_min_trials=1)  # https://optuna.readthedocs.io/en/stable/reference/pruners.html
  optuna_sampler = optuna.samplers.TPESampler(seed=SEED_GLOBAL, consider_prior=True, prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False, 
                                              n_startup_trials=N_STARTUP_TRIALS_SAMPLING, n_ei_candidates=24, multivariate=False, group=False, warn_independent_sampling=True, constant_liar=False)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler
  study = optuna.create_study(direction="maximize", study_name=None, pruner=optuna_pruner, sampler=optuna_sampler)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html

  study.optimize(lambda trial: optuna_objective(trial, lang=lang, n_sample=n_sample, df_train=df_train, df=df_cl),  # hypothesis_hyperparams_dic=hypothesis_hyperparams_dic
                n_trials=N_TRIALS, show_progress_bar=True)  # Objective function with additional arguments https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
  return study


hp_study_dic = {}
for n_sample in N_SAMPLE_DEV:
    hp_study_dic_scenario = {}
    # ! only sample of languages for faster testing
    for lang in tqdm.tqdm(LANGUAGES[:3], leave=False):

      if (lang == LANGUAGE_TRAIN) and (AUGMENTATION in ["no-nmt-single", "one2anchor", "one2many"]):
          # if train-lang is en, then no testing on en as target-lang necessary (?) - safer to keep in for now, can remove results later ?
          # also consider what this means for final runs
          continue

      study = run_study(n_sample=n_sample, lang=lang)
      print(f"Run for language {lang} finished.")

      if (AUGMENTATION in ["no-nmt-single", "one2many", "many2anchor"]) or ((AUGMENTATION in ["no-nmt-many", "many2many"]) and (VECTORIZER in ["embeddings-multi"])) or ((AUGMENTATION in ["one2anchor"]) and (VECTORIZER in ["tfidf", "embeddings-en"])):
          print("Study params: ", LANGUAGES, f" {AUGMENTATION}_{VECTORIZER}_{MODEL_NAME.split('/')[-1]}_{DATASET_NAME}_{n_sample}")
          print("Best study value: ", study.best_value)
          hp_study_dic_lang = {"language": LANGUAGES, "vectorizer": VECTORIZER, "augmentation": AUGMENTATION, "method": METHOD, "n_sample": n_sample, "dataset": DATASET_NAME, "algorithm": MODEL_NAME, "optuna_study": study}
          hp_study_dic_scenario.update(hp_study_dic_lang)
          break  # these runs do not need different hp-searches for different lang
      elif ((AUGMENTATION in ["no-nmt-many", "many2many"]) and (VECTORIZER != "embeddings-multi")) or ((AUGMENTATION in ["one2anchor"]) and (VECTORIZER in ["embeddings-multi"])):
          print("Study params: ", lang, f" {AUGMENTATION}_{VECTORIZER}_{MODEL_NAME.split('/')[-1]}_{DATASET_NAME}_{n_sample}")
          print("Best study value: ", study.best_value)
          hp_study_dic_lang = {lang: {"language": lang, "vectorizer": VECTORIZER, "augmentation": AUGMENTATION, "method": METHOD, "n_sample": n_sample, "dataset": DATASET_NAME, "algorithm": MODEL_NAME, "optuna_study": study} }
          hp_study_dic_scenario.update(hp_study_dic_lang)

    hp_study_dic_scenario = {f"{AUGMENTATION}_{VECTORIZER}_{MODEL_NAME.split('/')[-1]}_{DATASET_NAME}_{n_sample}": hp_study_dic_scenario}
    hp_study_dic.update(hp_study_dic_scenario)

    # harmonise length of n_sample string (always 5 characters)
    while len(str(n_sample)) <= 4:
        n_sample = "0" + str(n_sample)

    ## save studies
    if EXECUTION_TERMINAL == True:
      #if VECTORIZER == "tfidf":
      joblib.dump(hp_study_dic_scenario, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")
      #elif VECTORIZER == "embeddings":
      #  joblib.dump(hp_study_dic_scenario, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")

    elif EXECUTION_TERMINAL == False:
      #if VECTORIZER == "tfidf":
      joblib.dump(hp_study_dic_scenario, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}_t.pkl")
      #elif VECTORIZER == "embeddings":
      #  joblib.dump(hp_study_dic_scenario, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}_t.pkl")

## stop carbon tracker
#if CARBON_TRACKING:
#  tracker.stop()  # writes csv file to directory specified during initialisation. Does not overwrite csv, but append new runs

print("Run done.")

