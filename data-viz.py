#!/usr/bin/env python
# coding: utf-8

# ## Install and load packages
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import joblib

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

#metric = "f1_micro"  # for figure with performance per dataset


# ## Data loading
#set wd
print(os.getcwd())
if "multilingual-repo" not in os.getcwd():
    os.chdir("./multilingual-repo")
print(os.getcwd())

dataset_name = "manifesto-8"

experiment_dic = {}
path_dataset = f"./results/{dataset_name}"
file_names_lst = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f)) and "experiment" in f]
[experiment_dic.update({file_name: joblib.load(f"./results/{dataset_name}/{file_name}")}) for file_name in file_names_lst]


#### create scenario tables
## wrangle the data out of the dics
experiment_dic_cols = {}
for i, key_experiment in enumerate(experiment_dic.keys()):
    for information in experiment_dic[key_experiment]["experiment_summary"].keys():
        # create empty container for appending information. only necessary for first experimentd ic
        if i == 0:  experiment_dic_cols.update({information: []})
        experiment_dic_cols[information].append(experiment_dic[key_experiment]["experiment_summary"][information])
        if experiment_dic[key_experiment]["experiment_summary"][information] is np.nan:
            print(key_experiment)

# create df
df_exp = pd.DataFrame(experiment_dic_cols)

## sorting dataframe
# make variables categorical to enable custom sorting order
df_exp["vectorizer"] = pd.Categorical(df_exp['vectorizer'], ["tfidf", "embeddings-en", "embeddings-multi"])
df_exp["augmentation"] = pd.Categorical(df_exp['augmentation'], ["no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"])
df_exp["model_name"] = pd.Categorical(df_exp['model_name'], ["logistic", "microsoft/MiniLM-L12-H384-uncased", "microsoft/Multilingual-MiniLM-L12-H384"])
df_exp = df_exp.sort_values(by=["model_name", "vectorizer", "augmentation"])
# removing less interesting columns
df_exp = df_exp[['model_name', 'vectorizer', 'augmentation', 'sample_size',
                 'f1_macro_mean', 'f1_micro_mean', #'accuracy_balanced_mean',
                 'f1_macro_mean_cross_lang_std', 'f1_micro_mean_cross_lang_std', #'accuracy_balanced_mean_cross_lang_std',
                 ]]

# subset df for two scenarios
augmentation_multi = ["no-nmt-many", "many2anchor", "many2many"]
df_exp_multi = df_exp[df_exp.augmentation.isin(augmentation_multi)]
df_exp_mono = df_exp[~df_exp.augmentation.isin(augmentation_multi)]






