

import sys
if sys.stdin.isatty():
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (sys.stdin.isatty(): ", sys.stdin.isatty(), ")")


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
# arguments only for test script
parser.add_argument('-cvf', '--n_cross_val_final', type=int, default=3,
                    help='For how many different random samples should the algorithm be tested at a given sample size?')
parser.add_argument('-zeroshot', '--zeroshot', action='store_true',
                    help='Start training run with a zero-shot run')

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
                    help='Name of dataset. Can be one of: "manifesto-8" ')
parser.add_argument('-samp', '--sample_interval', type=int, nargs='+',
                    help='Interval of sample sizes to test.')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "nli", "standard_dl"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model. Must fit to "method" argument.')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings-en" or "embeddings-multi"')
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
  # parse args if not in terminal, but in script. adapt manually
  args = parser.parse_args(["--n_cross_val_final", "2",  #--zeroshot
                            "--dataset", "manifesto-8",
                            "--languages", "en", "de", "es", "fr", "tr", "ru", "ko",
                            "--language_anchor", "en", "--language_train", "en",  # in multiling scenario --language_train needs to be list of lang (?)
                            "--augmentation_nmt", "many2many",  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
                            "--sample_interval", "300",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "classical_ml", "--model", "microsoft/Multilingual-MiniLM-L12-H384",  # "microsoft/Multilingual-MiniLM-L12-H384", "microsoft/MiniLM-L12-H384-uncased"
                            "--vectorizer", "embeddings-multi",  # "tfidf", "embeddings-en", "embeddings-multi"
                            "--hyperparam_study_date", "20221026"])



### args only for test runs
CROSS_VALIDATION_REPETITIONS_FINAL = args.n_cross_val_final
ZEROSHOT = args.zeroshot

### args for both hyperparameter tuning and test runs
# choose dataset
DATASET_NAME = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
LANGUAGES = args.languages
LANGUAGES = LANGUAGES[:3]
LANGUAGE_TRAIN = args.language_train
LANGUAGE_ANCHOR = args.language_anchor
AUGMENTATION = args.augmentation_nmt

N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set
VECTORIZER = args.vectorizer

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "classical_ml"
MODEL_NAME = args.model  # "SVM"

HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"





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
    raise Exception(f"Dataset not defined: {DATASET_NAME}")

print(DATASET_NAME)




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

from helpers import compute_metrics_standard, clean_memory
from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer

## functions for scenario data selection and augmentation
from helpers import select_data_for_scenario_hp_search, select_data_for_scenario_final_test, data_augmentation, sample_for_scenario, choose_preprocessed_text




# ## Final test with best hyperparameters

## hyperparameters for final tests
# selective load one decent set of hps for testing
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results/manifesto-8/optuna_study_deberta-v3-base_01000samp_20221006.pkl")

# select best hp
n_sample = N_SAMPLE_DEV[0]
n_sample_string = N_SAMPLE_DEV[0]
while len(str(n_sample_string)) <= 4:
    n_sample_string = "0" + str(n_sample_string)

## testing with manually set hyperparameters
N_SAMPLE_TEST = N_SAMPLE_DEV * len(LANGUAGES)
HYPER_PARAMS_LST = [{'lr_scheduler_type': 'constant', 'learning_rate': 2e-5, 'num_train_epochs': 40, 'seed': 42, 'per_device_train_batch_size': 16, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128}]
HYPER_PARAMS_LST = HYPER_PARAMS_LST * len(LANGUAGES)
print(HYPER_PARAMS_LST)

# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa".lower() in MODEL_NAME.lower(): fp16_bool = False  # mDeBERTa does not support FP16 yet




#### K example intervals loop
# will automatically only run once if only 1 element in HYPER_PARAMS_LST for runs with one good hp-value
experiment_details_dic = {}
for lang, hyperparams in tqdm.tqdm(zip(LANGUAGES, HYPER_PARAMS_LST), desc="Iterations for different languages and hps", leave=True):
  np.random.seed(SEED_GLOBAL)
  t_start = time.time()   # log how long training of model takes

  ### select correct language scenario for train sampling and testing
  df_train_scenario, df_test_scenario = select_data_for_scenario_final_test(df_train=df_train, df_test=df_test, lang=lang, augmentation=AUGMENTATION, vectorizer=VECTORIZER, language_train=LANGUAGE_TRAIN, language_anchor=LANGUAGE_ANCHOR)


  # prepare loop
  k_samples_experiment_dic = {"method": METHOD, "language_source": lang, "language_anchor": LANGUAGE_ANCHOR, "n_sample": n_sample, "model": MODEL_NAME, "vectorizer": VECTORIZER, "augmentation": AUGMENTATION, "hyperparams": hyperparams}  # "trainer_args": train_args, "hypotheses": HYPOTHESIS_TYPE, "dataset_stats": dataset_stats_dic
  f1_macro_lst = []
  f1_micro_lst = []
  accuracy_balanced_lst = []
  # randomness stability loop. Objective: calculate F1 across N samples to test for influence of different (random) samples
  np.random.seed(SEED_GLOBAL)
  for random_seed_sample in tqdm.tqdm(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_FINAL), desc="Iterations for std", leave=True):

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

    if (n_sample == 0) and (METHOD == "standard_dl"):
      metric_step = {'accuracy_balanced': 0, 'accuracy_not_b': 0, 'f1_macro': 0, 'f1_micro': 0}
      f1_macro_lst.append(0)
      f1_micro_lst.append(0)
      accuracy_balanced_lst.append(0)
      k_samples_experiment_dic.update({"n_train_total": len(df_train_scenario_samp), f"metrics_seed_{random_seed_sample}": metric_step})
      break


    ### data augmentation on sample for multiling models + translation scenarios
    # general function - common with hp-search script
    df_train_scenario_samp_augment = data_augmentation(df_train_scenario_samp=df_train_scenario_samp, df_train=df_train, lang=lang, augmentation=AUGMENTATION, vectorizer=VECTORIZER, language_train=LANGUAGE_TRAIN, language_anchor=LANGUAGE_ANCHOR)

    print("Number of training examples after (potential) augmentation: ", len(df_train_scenario_samp_augment))


    ### text pre-processing
    # !? double check if it's correct to just always take the "text_original_trans" column for both BERT models. should be correct because contains original text if iso-original == iso-trans. This should be done correctly upstream
    df_train_scenario_samp_augment_textcol, df_test_scenario_textcol = choose_preprocessed_text(df_train_scenario_samp_augment=df_train_scenario_samp_augment, df_test_scenario=df_test_scenario, augmentation=AUGMENTATION, vectorizer=VECTORIZER, vectorizer_sklearn=None, language_train=LANGUAGE_TRAIN, language_anchor=LANGUAGE_ANCHOR, method=METHOD)
    df_train_scenario_samp_augment_textcol = df_train_scenario_samp_augment_textcol.sample(frac=1.0, random_state=random_seed_sample)

    clean_memory()
    model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
    encoded_dataset = tokenize_datasets(df_train_samp=df_train_scenario_samp_augment_textcol, df_test=df_test_scenario_textcol, tokenizer=tokenizer, method=METHOD, max_length=None)

    train_args = set_train_args(hyperparams_dic=hyperparams, training_directory=TRAINING_DIRECTORY, disable_tqdm=False, evaluation_strategy="no", fp16=fp16_bool)  # seed=random_seed_sample ! can the order to data loading via seed (but then different from HP search)
    trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=encoded_dataset, train_args=train_args, 
                             method=METHOD, label_text_alphabetical=LABEL_TEXT_ALPHABETICAL)
    clean_memory()


    if n_sample != 0:
      trainer.train()


    ### metrics
    results = trainer.evaluate()  # eval_dataset=encoded_dataset_test

    k_samples_experiment_dic.update({"n_train_total": len(df_train_scenario_samp), f"metrics_seed_{random_seed_sample}": results})
    f1_macro_lst.append(results["eval_f1_macro"])
    f1_micro_lst.append(results["eval_f1_micro"])
    accuracy_balanced_lst.append(results["eval_accuracy_balanced"])

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
  k_samples_experiment_dic.update({"metrics_mean": metrics_mean, "dataset": DATASET_NAME, "n_classes": len(LABEL_TEXT_ALPHABETICAL), "train_eval_time_per_model": t_total})


  # update of overall experiment dic
  experiment_details_dic_step = {f"experiment_sample_{n_sample_string}_{METHOD}_{MODEL_NAME}_{lang}": k_samples_experiment_dic}
  experiment_details_dic.update(experiment_details_dic_step)


  ## stop loop for multiple language case - no separate iterations per language necessary
  #if "many2anchor" in AUGMENTATION:
  #  break
  #if ("multi" in VECTORIZER) and (any(augmentation in AUGMENTATION for augmentation in ["no-nmt-many", "many2many"])):  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
  #  break

  ## save experiment dic after each new study
  #joblib.dump(experiment_details_dic_step, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{n_sample_string}samp_{HYPERPARAM_STUDY_DATE}.pkl")



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
  joblib.dump(experiment_details_dic, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{HYPERPARAM_STUDY_DATE}.pkl")
elif EXECUTION_TERMINAL == False:
  joblib.dump(experiment_details_dic, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{HYPERPARAM_STUDY_DATE}_t.pkl")


print("Run done.")




