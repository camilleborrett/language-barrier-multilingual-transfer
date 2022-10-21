#!/usr/bin/env python
# coding: utf-8

EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL)

# ! had protobuf==4.21.3, downgraded to pip install protobuf==3.20.* to avoid error when running minilm https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly
# need to add to requirements.txt if it works

# can maybe save disk space by deleting some cached models
from transformers import file_utils
print(file_utils.default_cache_path)

# ## Load packages
import transformers
import torch
import datasets
import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
import random
import os
import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from datetime import date
import time
import joblib

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
parser = argparse.ArgumentParser(description='Run hyperparameter tuning with different algorithms on different datasets')

## Add the arguments
# arguments for hyperparameter search
parser.add_argument('-context', '--context', action='store_true',
                    help='Take surrounding context sentences into account. Only use flag if context available.')

# arguments for both hyperparam and test script
parser.add_argument('-lang', '--languages', type=str, nargs='+',
                    help='List of languages to iterate over. e.g. "en", "de", "es", "fr", "ko", "ja", "tr", "ru" ')
parser.add_argument('-anchor', '--language_anchor', type=str,
                    help='Anchor language to translate all texts to if using anchor. Default is "en"')
parser.add_argument('-language_train', '--language_train', type=str,
                    help='What language should the training set be in?. Default is "en"')
parser.add_argument('-augment', '--augmentation_nmt', type=str,
                    help='Whether and how to augment the data with neutral machine translation (NMT).')

parser.add_argument('-ds', '--dataset', type=str,
                    help='Name of dataset. Can be one of: "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"')
parser.add_argument('-samp', '--sample_interval', type=int, nargs='+',
                    help='Interval of sample sizes to test.')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "nli", "standard_dl"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model. Must fit to "method" argument.')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings-en" or "embeddings-multi"')
parser.add_argument('-tqdm', '--disable_tqdm', action='store_true',
                    help='Adding the flag enables tqdm for progress tracking')
parser.add_argument('-carbon', '--carbon_tracking', action='store_true',
                    help='Adding the flag enables carbon tracking via CodeCarbon')  # not used, as CodeCarbon caused bugs https://github.com/mlco2/codecarbon/issues/305

parser.add_argument('-hp_date', '--hyperparam_study_date', type=str,
                    help='Date string to specifiy which hyperparameter run should be selected. e.g. "20220304"')

# arguments only for test script
parser.add_argument('-cvf', '--n_cross_val_final', type=int, default=3,
                    help='For how many different random samples should the algorithm be tested at a given sample size?')
parser.add_argument('-zeroshot', '--zeroshot', action='store_true',
                    help='Start training run with a zero-shot run')



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
  # parse args if not in terminal, but in script. adapt manually
  args = parser.parse_args(["--dataset", "manifesto-8",
                            "--languages", "en", "de", "es", "fr", "tr", "ru",   #"en", "de", "es", "fr", "tr", "ru", "ko", "ja"   # ! ko, ja excluded due to train-test-split issue
                            "--language_anchor", "en", "--language_train", "en",
                            "--augmentation_nmt", "no-nmt-single",  #"no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
                            "--sample_interval", "500",  # "100", "500", "1000", "2500", "5000", #"10000",
                            "--method", "standard_dl", "--model", "microsoft/Multilingual-MiniLM-L12-H384",  # "microsoft/Multilingual-MiniLM-L12-H384", "microsoft/MiniLM-L12-H384-uncased"
                            "--vectorizer", "transformer-multi",  # "tfidf", "embeddings-en", "embeddings-multi", "transformer-mono", "transformer-multi"
                            "--n_cross_val_final", "2", "--hyperparam_study_date", "20220725"])


### args for both hyperparameter tuning and test runs
# choose dataset
DATASET_NAME = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
LANGUAGES = args.languages
LANGUAGE_TRAIN = args.language_train
LANGUAGE_ANCHOR = args.language_anchor
AUGMENTATION = args.augmentation_nmt

N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "nsp", "classical_ml"
MODEL_NAME = args.model

DISABLE_TQDM = args.disable_tqdm
CARBON_TRACKING = args.carbon_tracking

### args only for test runs
HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"
CROSS_VALIDATION_REPETITIONS_FINAL = args.n_cross_val_final
ZEROSHOT = args.zeroshot
VECTORIZER = args.vectorizer



# ## Load data
if "manifesto-8" in DATASET_NAME:
  df_cl = pd.read_csv("./data-clean/df_manifesto_all.csv", index_col="idx")
  df_train = pd.read_csv("./data-clean/df_manifesto_train_trans_embed.csv", index_col="idx")
  df_test = pd.read_csv("./data-clean/df_manifesto_test_trans_embed.csv", index_col="idx")

elif DATASET_NAME == "manifesto-military":
  df_cl = pd.read_csv("./data-clean/df_manifesto_military_cl.csv", index_col="idx")
  df_train = pd.read_csv("./data-clean/df_manifesto_military_train.csv", index_col="idx")
  df_test = pd.read_csv("./data-clean/df_manifesto_military_test.csv", index_col="idx")
elif DATASET_NAME == "manifesto-protectionism":
  df_cl = pd.read_csv("./data-clean/df_manifesto_protectionism_cl.csv", index_col="idx")
  df_train = pd.read_csv("./data-clean/df_manifesto_protectionism_train.csv", index_col="idx")
  df_test = pd.read_csv("./data-clean/df_manifesto_protectionism_test.csv", index_col="idx")
elif DATASET_NAME == "manifesto-morality":
  df_cl = pd.read_csv("./data-clean/df_manifesto_morality_cl.csv", index_col="idx")
  df_train = pd.read_csv("./data-clean/df_manifesto_morality_train.csv", index_col="idx")
  df_test = pd.read_csv("./data-clean/df_manifesto_morality_test.csv", index_col="idx")
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
    raise Exception(f"Dataset not defined: {DATASET_NAME}")

print(DATASET_NAME)

## fill na for non translated texts to facilitate downstream analysis
df_train["language_iso_trans"] = [language_iso if pd.isna(language_iso_trans) else language_iso_trans for language_iso, language_iso_trans in zip(df_train.language_iso, df_train.language_iso_trans)]
df_train["text_original_trans"] = [text_original if pd.isna(text_original_trans) else text_original_trans for text_original, text_original_trans in zip(df_train.text_original, df_train.text_original_trans)]
df_test["language_iso_trans"] = [language_iso if pd.isna(language_iso_trans) else language_iso_trans for language_iso, language_iso_trans in zip(df_test.language_iso, df_test.language_iso_trans)]
df_test["text_original_trans"] = [text_original if pd.isna(text_original_trans) else text_original_trans for text_original, text_original_trans in zip(df_test.text_original, df_test.text_original_trans)]



## reduce max sample size interval list to fit to max df_train length
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample <= len(df_train)]
if len(df_train) < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len(df_train)]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)


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

from analysis import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)

from analysis.helpers import format_nli_testset, format_nli_trainset, data_preparation  # custom_train_test_split, custom_train_test_split_sent_overlapp
from analysis.helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer
from analysis.helpers import compute_metrics_standard, compute_metrics_nli_binary, compute_metrics_classical_ml, clean_memory

### load suitable hypotheses_hyperparameters and text formatting function
from analysis.hypothesis_hyperparams import hypothesis_hyperparams


### load the hypothesis hyperparameters for the respective dataset
hypothesis_hyperparams_dic, format_text = hypothesis_hyperparams(dataset_name=DATASET_NAME, df_cl=df_cl)

# check which template fits to standard_dl or NLI
print("")
print([template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" in template])
print([template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" not in template])

# in case -context flag is passed, but dataset actually does not have context
nli_templates = [template for template in list(hypothesis_hyperparams_dic.keys()) if "not_nli" not in template]
if "context" not in nli_templates:
  CONTEXT = False




# ## Final test with best hyperparameters

### parameters for final tests with best hyperparams
## load existing studies with hyperparams
# selective load one decent set of hps for testing
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results/manifesto-8/optuna_study_deberta-v3-base_00500samp_20220428.pkl")

## testing with manually set hyperparameters
N_SAMPLE_TEST = N_SAMPLE_DEV * len(LANGUAGES)
HYPOTHESIS_TEMPLATE_LST = ["template_not_nli"] * len(LANGUAGES)
HYPER_PARAMS_LST_TEST = [{'lr_scheduler_type': 'constant', 'learning_rate': 4e-5, 'num_train_epochs': 2, 'seed': 42, 'per_device_train_batch_size': 16, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128}]
HYPER_PARAMS_LST_TEST = HYPER_PARAMS_LST_TEST * len(LANGUAGES)
print(HYPER_PARAMS_LST_TEST)

"""hp_study_dic = {}
for n_sample in N_SAMPLE_DEV:   
  while len(str(n_sample)) <= 4:
    n_sample = "0" + str(n_sample)
  hp_study_dic_step = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")
  #hp_study_dic_step = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'.split('/')[-1]}_{n_sample}samp_20220220.pkl")
  hp_study_dic.update(hp_study_dic_step)"""

"""if ZEROSHOT == False:
  N_SAMPLE_TEST = N_SAMPLE_DEV * len(LANGUAGES)
  print(N_SAMPLE_TEST)

  HYPER_PARAMS_LST = [study_value['optuna_study'].best_trial.user_attrs["hyperparameters_all"] for study_key, study_value in hp_study_dic.items()]
  
  # hypothesis template: always simple without context and without NLI for this paper
  #HYPOTHESIS_TEMPLATE_LST = [hyperparams_dic["hypothesis_template"] for hyperparams_dic in HYPER_PARAMS_LST]  #if ("context" in hyperparams_dic["hypothesis_template"])
  #HYPOTHESIS_TEMPLATE_LST = HYPOTHESIS_TEMPLATE_LST * len(LANGUAGES)
  HYPOTHESIS_TEMPLATE_LST = ["template_not_nli"] * len(LANGUAGES)
  print(HYPOTHESIS_TEMPLATE_LST)

  HYPER_PARAMS_LST = [{key: dic[key] for key in dic if key!="hypothesis_template"} for dic in HYPER_PARAMS_LST]  # return dic with all elements, except hypothesis template
  HYPER_PARAMS_LST_TEST = HYPER_PARAMS_LST
  HYPER_PARAMS_LST_TEST = HYPER_PARAMS_LST_TEST * len(LANGUAGES)
  print(HYPER_PARAMS_LST_TEST)

else:
    raise Exception("zero-shot classification not implemented")"""

## for specific targeted runs
#N_SAMPLE_TEST = [N_SAMPLE_TEST[1]]
#HYPOTHESIS_TEMPLATE_LST = [HYPOTHESIS_TEMPLATE_LST[1]]
#HYPER_PARAMS_LST_TEST = [HYPER_PARAMS_LST_TEST[1]]
#print(HYPER_PARAMS_LST_TEST)

## for only 0-shot runs
"""if N_SAMPLE_DEV == [0]:
  N_SAMPLE_TEST = [0]
  HYPOTHESIS_TEMPLATE_LST = ["template_complex"]  # sentiment-news-econ: "template_complex", coronanet: "template_quote", cap-sotu: "template_quote_context", cap-us-court: "template_quote", manifesto-8: "template_quote_context",  manifesto-military: "template_quote_2", manifesto-morality: "template_quote_2", manifesto-protectionism: "template_quote_2"
  HYPER_PARAMS_LST_TEST = [{"per_device_eval_batch_size": 160}]
"""



## intermediate text formatting function for testing code on manifesto-8 without NLI and without context
def format_text(df=None, text_format=None, embeddings=VECTORIZER, translated_text=True):
    # ! translated_text is for df_test (false) or not df_train (true)
    # ! review 'translated_text' - still up to date with updated code? can be removed?
    #if (text_format == 'template_not_nli') and (translated_text == False) and (embeddings == "tfidf"):
    #    df["text_prepared"] = df.text_original
    #elif (text_format == 'template_not_nli') and (translated_text == True) and (embeddings == "tfidf"):
    #    df["text_prepared"] = df.text_original_trans
    if (text_format == 'template_not_nli') and (translated_text == False) and (embeddings == "transformer-mono"):
        df["text_prepared"] = df.text_original
    elif (text_format == 'template_not_nli') and (translated_text == True) and (embeddings == "transformer-mono"):
        df["text_prepared"] = df.text_original_trans
    elif (text_format == 'template_not_nli') and (embeddings == "transformer-multi"):
        df["text_prepared"] = df.text_original
    else:
        raise Exception(f'format_text did not work for text_format == {text_format}, vectorizer == {embeddings}, translated_text == {translated_text}')

    # ! special case for no-nmt-multi, don't have monolingual models - taking multilingual embeddings on single languages as proxy
    #if (text_format == 'template_not_nli') and (embeddings == "embeddings-en") and (AUGMENTATION == "no-nmt-many"):
    #    df["text_prepared"] = df.text_original_trans_embed_multi

    return df.copy(deep=True)


# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa" in MODEL_NAME: fp16_bool = False  # mDeBERTa does not support FP16 yet





### run random cross-validation for hyperparameter search without a dev set
np.random.seed(SEED_GLOBAL)

### K example intervals loop
experiment_details_dic = {}
for lang, n_max_sample, hyperparams, hypothesis_template in tqdm.tqdm(zip(LANGUAGES, N_SAMPLE_TEST, HYPER_PARAMS_LST_TEST, HYPOTHESIS_TEMPLATE_LST), desc="Iterations for different number of texts", leave=True):
  np.random.seed(SEED_GLOBAL)
  t_start = time.time()   # log how long training of model takes

  ### select correct language for train sampling and test
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
      if "multi" in VECTORIZER:
          df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
          df_test_lang = df_test.query("language_iso == language_iso_trans").copy(deep=True)
      elif "multi" not in VECTORIZER:
          raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")
  else:
      raise Exception("Issue with AUGMENTATION")


  # prepare loop
  k_samples_experiment_dic = {"method": METHOD, "language_source": lang, "language_anchor": LANGUAGE_ANCHOR, "n_max_sample": n_max_sample, "model": MODEL_NAME, "vectorizer": VECTORIZER, "augmentation": AUGMENTATION, "hyperparams": hyperparams}  # "trainer_args": train_args, "hypotheses": HYPOTHESIS_TYPE, "dataset_stats": dataset_stats_dic
  f1_macro_lst = []
  f1_micro_lst = []
  accuracy_balanced_lst = []
  # randomness stability loop. Objective: calculate F1 across N samples to test for influence of different (random) samples
  for random_seed_sample in tqdm.tqdm(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_FINAL), desc="Iterations for std", leave=True):

    ## sampling
    if n_max_sample == 999_999:  # all data, no sampling
      df_train_samp = df_train_lang.copy(deep=True)
    # same sample size per language for multiple language data scenarios
    elif any(augmentation in AUGMENTATION for augmentation in ["no-nmt-many", "many2anchor", "many2many"]):
        df_train_samp = df_train_lang.groupby(by="language_iso").apply(lambda x: x.sample(n=min(n_max_sample, len(x)), random_state=random_seed_sample).copy(deep=True))
    # one sample size for single language data scenario
    else:
      df_train_samp = df_train_lang.sample(n=min(n_max_sample, len(df_train_lang)), random_state=random_seed_sample).copy(deep=True)
    print("Number of training examples after sampling: ", len(df_train_samp), " . (but before cross-validation split) ")

    if (n_max_sample == 0) and (METHOD == "standard_dl"):
      metric_step = {'accuracy_balanced': 0, 'accuracy_not_b': 0, 'f1_macro': 0, 'f1_micro': 0}
      f1_macro_lst.append(0)
      f1_micro_lst.append(0)
      accuracy_balanced_lst.append(0)
      k_samples_experiment_dic.update({"n_train_total": len(df_train_samp), f"metrics_seed_{random_seed_sample}": metric_step})
      break


    ### data augmentation on sample
    ## single language text scenarios
    sample_sent_id = df_train_samp.sentence_id.unique()
    if AUGMENTATION == "no-nmt-single":
        df_train_samp_augment = df_train_samp.copy(deep=True)
    elif AUGMENTATION == "one2anchor":
        if "multi" not in VECTORIZER:
            df_train_samp_augment = df_train_samp.copy(deep=True)
        elif "multi" in VECTORIZER:
            # augment by combining texts from anchor language with texts from anchor translated to target language
            df_train_augment = df_train[(((df_train.language_iso == LANGUAGE_TRAIN) & (df_train.language_iso_trans == LANGUAGE_ANCHOR)) | ((df_train.language_iso == LANGUAGE_TRAIN) & (df_train.language_iso_trans == lang)))].copy(deep=True)
            df_train_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
        else:
            raise Exception("Issue with VECTORIZER")
    elif AUGMENTATION == "one2many":
        if "multi" in VECTORIZER:
            df_train_augment = df_train.query("language_iso == @LANGUAGE_TRAIN").copy(deep=True)  # can use all translations and original text (for single train lang) here
            df_train_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
        else:
            raise Exception("AUGMENTATION == 'X2many' only works for multilingual vectorization")
    ## multiple language text scenarios
    elif AUGMENTATION == "no-nmt-many":
        df_train_samp_augment = df_train_samp.copy(deep=True)
    elif AUGMENTATION == "many2anchor":
        if "multi" not in VECTORIZER:
            df_train_samp_augment = df_train_samp.copy(deep=True)
        if "multi" in VECTORIZER:
            # already have all original languages in the scenario. augmenting it with translated (to anchor) texts here. e.g. for 7*6=3500 original texts, adding 6*6=3000 texts, all translated to anchor (except anchor texts)
            df_train_augment = df_train[(df_train.language_iso == df_train.language_iso_trans) | (df_train.language_iso_trans == LANGUAGE_ANCHOR)].copy(deep=True)
            df_train_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
    elif AUGMENTATION == "many2many":
        if "multi" in VECTORIZER:
            # already have all original languages in the scenario. augmenting it with all other translated texts
            df_train_augment = df_train.copy(deep=True)
            df_train_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
        else:
            raise Exception(f"AUGMENTATION == {AUGMENTATION} only works for multilingual vectorization")


    # chose the text format depending on hyperparams
    # returns "text_prepared" column, e.g. with concatenated sentences
    df_train_samp_augment = format_text(df=df_train_samp_augment, text_format=hypothesis_template, embeddings=VECTORIZER, translated_text=False)
    df_train_samp_augment = df_train_samp_augment.sample(frac=1.0, random_state=random_seed_sample)  # shuffle df_train
    df_test_formatted = format_text(df=df_test_lang, text_format=hypothesis_template, embeddings=VECTORIZER, translated_text=True)

    # format train and dev set for NLI etc.
    #if METHOD == "nli":
    #  df_train_samp = format_nli_trainset(df_train=df_train_samp_augment, hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template], random_seed=SEED_GLOBAL)
    #  df_test_formatted = format_nli_testset(df_test=df_test_formatted, hypo_label_dic=hypothesis_hyperparams_dic[hypothesis_template])  # hypo_label_dic_short , hypo_label_dic_long

    clean_memory()
    model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
    encoded_dataset = tokenize_datasets(df_train_samp=df_train_samp_augment, df_test=df_test_formatted, tokenizer=tokenizer, method=METHOD, max_length=None)

    train_args = set_train_args(hyperparams_dic=hyperparams, training_directory=TRAINING_DIRECTORY, disable_tqdm=DISABLE_TQDM, evaluation_strategy="no", fp16=fp16_bool)  # seed=random_seed_sample ! can the order to data loading via seed (but then different from HP search)
    trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=encoded_dataset, train_args=train_args, 
                             method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
    clean_memory()

    if n_max_sample != 0:
      trainer.train()

    # metrics
    results = trainer.evaluate()  # eval_dataset=encoded_dataset_test

    k_samples_experiment_dic.update({"n_train_total": len(df_train_samp), f"metrics_seed_{random_seed_sample}": results})
    f1_macro_lst.append(results["eval_f1_macro"])
    f1_micro_lst.append(results["eval_f1_micro"])
    accuracy_balanced_lst.append(results["eval_accuracy_balanced"])

    if (n_max_sample == 0) or (n_max_sample == 999_999):  # only one inference necessary on same test set in case of zero-shot or full dataset
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
  k_samples_experiment_dic.update({"metrics_mean": metrics_mean, "dataset": DATASET_NAME, "n_classes": len(LABEL_TEXT_ALPHABETICAL), "train_eval_time_per_model": t_total})

  # harmonise n_sample file title
  while len(str(n_max_sample)) <= 4:
    n_max_sample = "0" + str(n_max_sample)

  # update of overall experiment dic
  experiment_details_dic_step = {f"experiment_sample_{n_max_sample}_{METHOD}_{MODEL_NAME}_{lang}": k_samples_experiment_dic}
  experiment_details_dic.update(experiment_details_dic_step)


  ## stop loop for multiple language case - no separate iterations per language necessary
  if "many2anchor" in AUGMENTATION:
    break
  if ("multi" in VECTORIZER) and (any(augmentation in AUGMENTATION for augmentation in ["no-nmt-many", "many2many"])):  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
    break

  ## save experiment dic after each new study
  #joblib.dump(experiment_details_dic_step, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{n_max_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")



### summary dictionary across all languages
experiment_summary_dic = {"experiment_summary":
     {"dataset": DATASET_NAME, "sample_size": N_SAMPLE_DEV, "method": METHOD, "model_name": MODEL_NAME, "vectorizer": VECTORIZER, "lang_anchor": LANGUAGE_ANCHOR, "lang_train": LANGUAGE_TRAIN, "lang_all": LANGUAGES, "augmentation": AUGMENTATION}
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
f1_macro_lst_mean = np.mean(f1_macro_lst_mean)
f1_micro_lst_mean = np.mean(f1_micro_lst_mean)
accuracy_balanced_lst_mean = np.mean(accuracy_balanced_lst_mean)
f1_macro_lst_mean_std = np.mean(f1_macro_lst_mean_std)
f1_micro_lst_mean_std = np.mean(f1_micro_lst_mean_std)
accuracy_balanced_lst_mean_std = np.mean(accuracy_balanced_lst_mean_std)

experiment_summary_dic["experiment_summary"].update({"f1_macro_mean": f1_macro_lst_mean, "f1_micro_mean": f1_micro_lst_mean, "accuracy_balanced_mean": accuracy_balanced_lst_mean,
                               "f1_macro_mean_std": f1_macro_lst_mean_std, "f1_micro_mean_std": f1_micro_lst_mean_std, "accuracy_balanced_mean_std": accuracy_balanced_lst_mean_std})

print(experiment_summary_dic)


### save full experiment dic
# merge individual languages experiments with summary dic
experiment_details_dic = {**experiment_details_dic, **experiment_summary_dic}


if EXECUTION_TERMINAL == True:
  joblib.dump(experiment_details_dic, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_max_sample}samp_{AUGMENTATION}_{HYPERPARAM_STUDY_DATE}.pkl")
elif EXECUTION_TERMINAL == False:
  joblib.dump(experiment_details_dic, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_max_sample}samp_{AUGMENTATION}_{HYPERPARAM_STUDY_DATE}_t.pkl")


print("Run done.")




