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

dataset_name_lst = ["manifesto-8", "pimpo_samp_a1"]

experiment_dic = {}
for dataset_name in dataset_name_lst:
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
df_results = pd.DataFrame(experiment_dic_cols)

## sorting dataframe
# make variables categorical to enable custom sorting order
df_results["vectorizer"] = pd.Categorical(df_results['vectorizer'], ["tfidf", "embeddings-en", "embeddings-multi"])
df_results["augmentation"] = pd.Categorical(df_results['augmentation'], ["no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"])
df_results["model_name"] = pd.Categorical(df_results['model_name'], ["logistic", "microsoft/deberta-v3-base", "microsoft/mdeberta-v3-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"])
df_results["nmt_model"] = pd.Categorical(df_results['nmt_model'], ["m2m_100_418M", "m2m_100_1.2B"])
df_results['method'] = ["classical+embed" if method == "classical_ml" and vectorizer != "tfidf" else method for method, vectorizer in zip(df_results['method'], df_results["vectorizer"])]
df_results["method"] = pd.Categorical(df_results['method'], ["classical_ml", "classical+embed", "standard_dl", "nli"])
df_results = df_results.sort_values(by=["dataset", "model_name", "method", "vectorizer", "augmentation", "nmt_model", "f1_macro_mean"], ascending=False)

# remove nmt_model column - can unclude again if both datasets have run with 1.2B
df_results = df_results[df_results.nmt_model == "m2m_100_418M"]

# cleaning and renaming
method_map = {
    "classical_ml": "Logistic Regression", "classical+embed": "Sent-Transformer + Log.Reg.",
    "standard_dl": "Transformer-base", "nli": "Transformer-NLI"
}
df_results["algorithm"] = df_results.method.map(method_map)
vectorizer_map = {
    "embeddings-multi": "multiling-embeddings", "embeddings-en": "monoling-embeddings", "tfidf": "tfidf"
}
df_results["representation"] = df_results.vectorizer.map(vectorizer_map)
augmentation_map = {
    "no-nmt-single": "no-mt-mono", "no-nmt-many": "no-mt-multi", "one2anchor": "one2anchor", "one2many": "one2many", "many2anchor": "many2anchor", "many2many": "many2many"
}
df_results["augmentation"] = df_results.augmentation.map(augmentation_map)
dataset_map = {
    "pimpo_samp_a1": "pimpo", "manifesto-8": "manifesto-8"
}
df_results["dataset"] = df_results.dataset.map(dataset_map)

# removing less interesting columns
df_results = df_results[["dataset", "algorithm", "representation", 'augmentation', #'sample_size', 'nmt_model',  # 'model_name', 'method', 'vectorizer'
                 'f1_macro_mean', 'f1_micro_mean', #'accuracy_balanced_mean',
                 'f1_macro_mean_cross_lang_std', 'f1_micro_mean_cross_lang_std', #'accuracy_balanced_mean_cross_lang_std',
                 ]]

df_results["f1_macro_mean"] = df_results["f1_macro_mean"].round(2)
df_results["f1_micro_mean"] = df_results["f1_micro_mean"].round(2)
df_results["f1_macro_mean_cross_lang_std"] = df_results["f1_macro_mean_cross_lang_std"].round(3)
df_results["f1_micro_mean_cross_lang_std"] = df_results["f1_micro_mean_cross_lang_std"].round(3)
df_results = df_results.rename(columns={"f1_macro_mean": "F1 Macro", "f1_micro_mean": "F1 Micro",
                                        "f1_macro_mean_cross_lang_std": "F1 Macro Lang. Std.",
                                        "f1_micro_mean_cross_lang_std": "F1 Micro Lang. Std.",
                                        "representation": "language representation",
                                        "augmentation": "MT augmentation"})


# subset df for two scenarios
augmentation_multi = ["no-mt-multi", "many2anchor", "many2many"]
df_results_multi = df_results[df_results["MT augmentation"].isin(augmentation_multi)]
df_results_mono = df_results[~df_results["MT augmentation"].isin(augmentation_multi)]

## reformat to display two datasets in separate columns, not rows
df_results_multi_manifesto = df_results_multi[df_results_multi.dataset == "manifesto-8"].reset_index(drop=True)
df_results_multi_pimpo = df_results_multi[df_results_multi.dataset == "pimpo"].reset_index(drop=True)
df_results_mono_manifesto = df_results_mono[df_results_mono.dataset == "manifesto-8"].reset_index(drop=True)
df_results_mono_pimpo = df_results_mono[df_results_mono.dataset == "pimpo"].reset_index(drop=True)
# join by columns
keys_join = ["algorithm", "language representation", "MT augmentation"]
df_results_mono = pd.merge(df_results_mono_manifesto, df_results_mono_pimpo, how="left", left_on=keys_join, right_on=keys_join, suffixes=[" (ma)", " (pi)"])
df_results_mono = df_results_mono[[col for col in df_results_mono.columns if "dataset" not in col]]  # remove dataset columns
df_results_multi = pd.merge(df_results_multi_manifesto, df_results_multi_pimpo, how="left", left_on=keys_join, right_on=keys_join, suffixes=[" (ma)", " (pi)"])
df_results_multi = df_results_multi[[col for col in df_results_multi.columns if "dataset" not in col]]  # remove dataset columns


# write to disk
df_results_mono.to_excel("./results/tables/df_results_mono_500samp.xlsx")
df_results_multi.to_excel("./results/tables/df_results_multi_500samp.xlsx")



#### Analysis which factor improves performance the most
# correlate performance with different methods
from scipy.stats import spearmanr
#pearsonr(df_results.method, df_results.corr_labels_avg)
df_results_corr = df_results.copy(deep=True)
df_results_corr["algorithm_factor"] = df_results_corr["algorithm"].factorize(sort=True)[0]
df_results_corr["representation_factor"] = df_results_corr["language representation"].factorize(sort=True)[0]
df_results_corr["augmentation_factor"] = df_results_corr["MT augmentation"].factorize(sort=True)[0]
#df_results_corr["nmt_model_factor"] = df_results_corr["nmt_model"].factorize(sort=True)[0]

#corr_spearman = spearmanr(df_results_corr[["method_factor", "size_factor", "vectorizer_factor", "languages_factor", "corr_labels_avg"]])  # , df_results_corr.corr_labels_avg
df_results_corr = df_results_corr[["algorithm_factor", "representation_factor", "augmentation_factor", "F1 Macro", "F1 Micro", "F1 Macro Lang. Std.", "F1 Micro Lang. Std."]]
df_corr_spearman = df_results_corr.corr(method="spearman")
# add p-value / significance
#rho = df.corr()
df_corr_pval = df_corr_spearman.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*df_corr_spearman.shape)
df_corr_pval = df_corr_pval.applymap(lambda x: ''.join(['*' for t in [0.001, 0.01, 0.05] if x <= t]))
df_results_corr = df_corr_spearman.round(2).astype(str) + df_corr_pval
df_results = df_results.round(2)

# write to disk
df_results_corr.to_excel("./results/tables/df_results_corr_500samp.xlsx")


print("Run done.")



