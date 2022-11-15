
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def load_data(languages=None, task=None, method=None, model_size=None, hypothesis=None, vectorizer=None, max_sample_lang=None, date=None):
    language_str = "_".join(languages)
    df = pd.read_csv(f"./results/pimpo/df_pimpo_pred_{task}_{method}_{model_size}_{hypothesis}_{vectorizer}_{max_sample_lang}samp_{language_str}_{date}.zip",
                     usecols=lambda x: x not in ["text_concat", "text_concat_embed_multi", "text_trans_concat", "text_trans_concat_embed_en", "text_trans_concat_tfidf",
                                                 "selection", "certainty_selection", "topic", "certainty_topic", "direction", "certainty_direction", "rn", "cmp_code"])
    # !? issue with NLI list in label column for nli-en-long
    # next run maybe fixes it because changed deep copy in format_nli_testset
    # intermediate solution
    df["label"] = pd.factorize(df["label_text"], sort=True)[0]

    df = df.rename(columns={"label_pred_text": "label_text_pred"}) # now fixed in new script
    # making date-month string to decate string
    df["decade"] = df["date"].apply(lambda x: str(x)[:3] + "0" if int(str(x)[3]) < 5 else str(x)[:3] + "5")

    ## for test data cleaning
    df = df[df.parfam_text != "AGR"]

    ## remove rows which were in training data
    df_train = df[df.label_text_pred.isna()]
    print(f"{len(df_train)} texts out of {len(df)} were in training data for this scenario. Excluding them for downstream analysis.")
    print(f"{len(df) - len(df_train)} of the population remain for tests.")
    df = df[~df.label_text_pred.isna()]

    # prediction only for languages not in training data
    df_cl = df[~df.language_iso.isin(languages)]

    return df_cl


#df_cl = load_data()


#### Calculate predicted or true distribution by meta data
def calculate_distribution(df_func=None, df_label_col="label_text_pred", exclude_no_topic=None, normalize=None, meta_data=None):
    # no-topic prediction irrelevant for the substantive task
    if exclude_no_topic:
        df_func = df_func[df_func[df_label_col] != "no_topic"]
        # df_cl = df.copy(deep=True)

    df_viz_counts = df_func.groupby(by=meta_data, as_index=True, group_keys=True).apply(lambda x: x[df_label_col].value_counts(normalize=normalize))
    df_viz_counts = df_viz_counts.reset_index().rename(columns={df_label_col: "label_count"}).rename(columns={"level_1": "label_text"})

    # harmonise df format to fix random bug
    n_labels = len(df_func[df_label_col].unique())
    #if len(df_viz_counts) == n_labels + 1:
    # !! double check if the columns get properly aligned here - will notice if results systematically wrong
    if all([label in df_viz_counts.columns for label in df_func[df_label_col].unique()]):
        meta_data_col = df_viz_counts[meta_data].tolist() * n_labels
        label_col = df_viz_counts.columns[1:n_labels+1].tolist()
        label_col = [[label] * len(df_viz_counts[meta_data].tolist()) for label in label_col]  #label_col * len(df_viz_counts[meta_data].tolist())  #* n_labels
        label_col = [item for sublist in label_col for item in sublist]
        label_values_col = [df_viz_counts[col].tolist() for col in np.unique(label_col)]
        label_values_col = [item for sublist in label_values_col for item in sublist]
        df_viz_counts = pd.DataFrame({meta_data: meta_data_col, "label_text": label_col, "label_count": label_values_col})

    # enforce ordering of columns to get custom column order
    if meta_data == "parfam_text":
        # ! add AGR here again later (both in pre and true)
        parfam_paper_order = ["ECO", "ETH", "LEF", "CHR", "LIB", "SOC", "CON", "NAT"]  # with AGR ["ECO", "ETH", "AGR", "LEF", "CHR", "LIB", "SOC", "CON", "NAT"]
        parfam_lr_order = ["ECO", "LEF", "SOC", "LIB", "CHR", "CON", "NAT", "ETH"]  # with AGR ["ECO", "LEF", "SOC", "LIB", "CHR", "CON", "NAT", "AGR", "ETH"]
        df_viz_counts[meta_data] = pd.Categorical(df_viz_counts[meta_data], parfam_lr_order)
        df_viz_counts = df_viz_counts.sort_values([meta_data, "label_text"])
    else:
        df_viz_counts = df_viz_counts.sort_values([meta_data, "label_text"])

    df_viz_counts = df_viz_counts[~df_viz_counts[meta_data].isna()]  # remove parfam (SIP) which is not in paper / categorical above

    return df_viz_counts

#df_viz_pred_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred")
#df_viz_true_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text")



##### calculating correlation
from scipy.stats import pearsonr

def compute_correlation(df_viz_true_counts=None, df_viz_pred_counts=None, meta_data=None, df=None):
    # remove no-topic predictions from df_true_counts if it was also previously removed from df_pred_counts
    # avoids downstream issues
    # ! but check that it does not introduce unintended behaviour
    if ("no_topic" in df_viz_true_counts.label_text.unique()) and ("no_topic" not in df_viz_pred_counts.label_text.unique()):
        df_viz_true_counts = df_viz_true_counts[df_viz_true_counts.label_text != "no_topic"]
    print("\n")

    ## add 0 if pred or true has value for meta-label combination which other df does not have
    # e.g. ground truth does not have specific label for specific country, but pred predicted label for this country
    df_merge = pd.merge(df_viz_true_counts, df_viz_pred_counts, how='outer', on=["label_text", meta_data], suffixes=("_true", "_pred")).fillna(0)
    df_merge = df_merge.sort_values(by=[meta_data, "label_text"])

    corr_dic = {}
    pearson_cor = pearsonr(df_merge.label_count_pred, df_merge.label_count_true)
    print("Pearson R overall: ", pearson_cor)
    corr_dic.update({"corr_labels_all": pearson_cor[0], f"p-value-labels_all": pearson_cor[1]})  # f"p-value-labels_all": pearson_cor[1]

    pearson_cor_mean = []
    for label in df.label_text.unique():
        if label != "no_topic":
            pearson_cor = pearsonr(df_merge[df_merge.label_text == label].label_count_pred, df_merge[df_merge.label_text == label].label_count_true)
            print(f"Person R for {label}: ", pearson_cor)
            corr_dic.update({f"corr_{label}": pearson_cor[0], f"p-value-{label}": pearson_cor[1]})  # f"p-value-{label}": pearson_cor[1]
            pearson_cor_mean.append(pearson_cor[0])
    pearson_cor_mean = np.mean(pearson_cor_mean)
    corr_dic.update({"corr_labels_avg": pearson_cor_mean})
    print(f"Person R label average: ", pearson_cor_mean)

    return corr_dic, df_merge

#corr_dic = compute_correlation(df_viz_true_counts=df_viz_true_counts, df_viz_pred_counts=df_viz_pred_counts)


### calculating classical ml metrics
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np

def compute_metrics_standard(label_gold=None, label_pred=None):
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_gold, label_pred, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_gold, label_pred, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    #acc_balanced = balanced_accuracy_score(label_gold, label_pred)
    acc_not_balanced = accuracy_score(label_gold, label_pred)
    metrics = {'f1_macro': f1_macro,
               #'f1_micro': f1_micro,
               #'accuracy_balanced': acc_balanced,
               'accuracy': acc_not_balanced,
               'precision_macro': precision_macro,
               'recall_macro': recall_macro,
               'precision_micro': precision_micro,
               'recall_micro': recall_micro,
               #'label_gold_raw': label_gold,
               #'label_predicted_raw': label_pred
               }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]})  # print metrics but without label lists
    #print("Detailed metrics: ", classification_report(label_gold, label_pred, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2,
    #                            output_dict=True,zero_division='warn'), "\n")

    return metrics

# ! computing labels for all 4 classes here including no-topic (on 3 not possible, because different length)
#metrics_dic = compute_metrics_standard(label_gold=df_cl.label, label_pred=df_cl.label_pred)
