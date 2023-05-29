#!/usr/bin/env python
# coding: utf-8

# ## Install and load packages
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import joblib
from sklearn.metrics import classification_report

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
experiment_dic_cols.update({"class_report": []})
[experiment_dic_cols.update({class_col: []}) for class_col in ["0", "1", "2", "3", "4", "5", "6", "7"]]

for i, key_experiment in enumerate(experiment_dic.keys()):
    for information in experiment_dic[key_experiment]["experiment_summary"].keys():
        # create empty container for appending information. only necessary for first experimentd ic
        if i == 0:  experiment_dic_cols.update({information: []})
        experiment_dic_cols[information].append(experiment_dic[key_experiment]["experiment_summary"][information])
        if experiment_dic[key_experiment]["experiment_summary"][information] is np.nan:
            print(key_experiment)

    # extract raw predictions for each experiment
    class_report_avg_experiment_lang_lst = []
    for experiment_per_lang in experiment_dic[key_experiment].keys():
        if experiment_per_lang != "experiment_summary":
            #experiment_dic_cols["class_report"].update([])
            class_report_lst = []
            # search over any key that contains "metrics_seed"
            for key in experiment_dic[key_experiment][experiment_per_lang].keys():
                if "metrics_seed" in key:
                    label_gold = experiment_dic[key_experiment][experiment_per_lang][key]["eval_label_gold_raw"]
                    label_pred = experiment_dic[key_experiment][experiment_per_lang][key]["eval_label_predicted_raw"]

                    class_report = classification_report(
                        label_gold, label_pred, #labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical,
                        digits=2, output_dict=True, zero_division=0
                    )
                    class_report_lst.append(class_report)
                    # only use first random seed for speed testing
                    #break

            # calculate the average for each class per random seed experiment
            class_report_avg_experiment = {}
            for key in class_report_lst[0].keys():
                if key not in ["accuracy", "macro avg", "weighted avg"]:
                    class_report_avg_experiment.update({key: np.mean([class_report[key]["f1-score"] for class_report in class_report_lst])})
            class_report_avg_experiment_lang_lst.append(class_report_avg_experiment)

    # calculate the average for each class per language experiment
    class_report_avg_experiment_lang = {}
    for key in class_report_avg_experiment_lang_lst[0].keys():
        class_report_avg_experiment_lang.update({key: np.mean([class_report[key] for class_report in class_report_avg_experiment_lang_lst])})

    experiment_dic_cols["class_report"].append(class_report_avg_experiment_lang)

    # classes as individual columns
    for key_class_all in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        #for key_class_real in class_report_avg_experiment_lang.keys():
        if key_class_all in class_report_avg_experiment_lang.keys():
            experiment_dic_cols[key_class_all].append(class_report_avg_experiment_lang[key_class_all])
        else:
            experiment_dic_cols[key_class_all].append(np.nan)


df_results_per_class = pd.DataFrame(experiment_dic_cols)

# sort
df_results_per_class["method"] = pd.Categorical(df_results_per_class['method'], ["classical_ml", "classical+embed", "standard_dl", "nli"])
df_results_per_class["augmentation"] = pd.Categorical(df_results_per_class['augmentation'], ['no-nmt-single', 'no-nmt-many', 'one2anchor', 'one2many', 'many2anchor', 'many2many', ])
df_results_per_class["vectorizer"] = pd.Categorical(df_results_per_class['vectorizer'], ["tfidf", "embeddings-en", "embeddings-multi"])

df_results_per_class = df_results_per_class.sort_values(by=["dataset", "method", "vectorizer", "augmentation"], ascending=False)


df_results_per_class_manifesto = df_results_per_class[df_results_per_class["dataset"] == "manifesto-8"]
df_results_per_class_pimpo = df_results_per_class[df_results_per_class["dataset"] == "pimpo_samp_a1"]

df_results_per_class_manifesto = df_results_per_class_manifesto[[
        '0', '1', '2', '3', '4', '5', '6', '7',
       'vectorizer', 'augmentation',  #'lang_train',
        'method',  #'f1_macro_mean', 'lang_anchor', 'lang_all',
       #'f1_micro_mean', 'accuracy_balanced_mean', # 'model_name', 'sample_size', 'nmt_model',
       'f1_macro_mean_cross_lang_std', #'f1_micro_mean_cross_lang_std',
       #'accuracy_balanced_mean_cross_lang_std', 'f1_macro_mean_std_mean',
       #'f1_micro_mean_std_mean', 'accuracy_balanced_mean_std_mean'
]]

label_num = [0, 1, 2, 3, 4, 5, 6, 7]
label_text = ['Economy', 'External Relations', 'Fabric of Society', 'Freedom and Democracy', 'No other category applies', 'Political System', 'Social Groups', 'Welfare and Quality of Life']
label_text_map_manifesto = {label_num[i]: label_text[i] for i in range(len(label_num))}
# !!  column-label mapping manually corrected in appendix: with NLI models, label 7 is "No other category applies". manually changed it to 4 to align with other models. the other labels are correctly mapped.

df_results_per_class_pimpo = df_results_per_class_pimpo[[
        '0', '1', '2', '3', #'4', '5', '6', '7',
       'vectorizer', 'augmentation',  #'lang_train',
        'method',  #'f1_macro_mean', 'lang_anchor', 'lang_all',
       #'f1_micro_mean', 'accuracy_balanced_mean', #'model_name', 'sample_size', 'nmt_model',
       'f1_macro_mean_cross_lang_std', #'f1_micro_mean_cross_lang_std',
       #'accuracy_balanced_mean_cross_lang_std', 'f1_macro_mean_std_mean',
       #'f1_micro_mean_std_mean', 'accuracy_balanced_mean_std_mean'
]]

label_num = [0, 1, 2, 3]
label_text = ['immigration_neutral', 'immigration_sceptical', 'immigration_supportive', 'no_topic']
label_text_map_pimpo = {label_num[i]: label_text[i] for i in range(len(label_num))}



# write to disk
df_results_per_class_manifesto = df_results_per_class_manifesto.round(2)
df_results_per_class_pimpo = df_results_per_class_pimpo.round(2)

#df_results_per_class_manifesto.to_excel("./results/viz-a1/df_results_per_class_manifesto.xlsx")
#df_results_per_class_pimpo.to_excel("./results/viz-a1/df_results_per_class_pimpo.xlsx")
