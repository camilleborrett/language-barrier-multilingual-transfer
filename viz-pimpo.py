

import pandas as pd
import plotly.graph_objects as go
import numpy as np

languages_lst = ["en"]
languages_lst = ["en", "de"]
languages_lst = ["en", "de", "sv", "fr"]
language_str = "_".join(languages_lst)

DATE = "221111"
TASK = "immigration"  # "immigration", "integration
METHOD = "nli"  # standard_dl nli
VECTORIZER = "en"
HYPOTHESIS = "long"
MAX_SAMPLE_LANG = "50"

META_DATA = "parfam_text"  # "parfam_text", "country_iso", "language_iso", "decade"
NORMALIZE = True


df = pd.read_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{language_str}_{DATE}.zip")
# !? issue with NLI list in label column for nli-en-long
# next run maybe fixes it because changed deep copy in format_nli_testset

df = df.rename(columns={"label_pred_text": "label_text_pred"}) # now fixed in new script




## adding date variable to predicted dfs via doc_id key
"""df_pimpo_samp = pd.read_csv("./data-clean/df_pimpo_samp.csv")
df_pimpo_samp_date = df_pimpo_samp[["doc_id", "date"]]
df_pimpo_samp_date = df_pimpo_samp_date[~df_pimpo_samp_date.doc_id.duplicated(keep="first")]
df_pimpo_samp_date["doc_id"] = df_pimpo_samp_date.doc_id.astype(str)
df["doc_id"] = df.doc_id.astype(str)
# merge
df = pd.merge(df, df_pimpo_samp_date, left_on="doc_id", right_on="doc_id", how="left")"""
# making date-month string to decate string
df["decade"] =  df["date"].apply(lambda x: str(x)[:3] + "0")




### 1. predicted distribution
# no-topic prediction irrelevant for the substantive task
df_viz_pred = df[df.label_text_pred != "no_topic"]
# prediction only for languages not in training data
df_viz_pred = df_viz_pred[~df_viz_pred.language_iso.isin(languages_lst)]

df_viz_pred_counts = df_viz_pred.groupby(by=META_DATA, as_index=True, group_keys=True).apply(lambda x: x.label_text_pred.value_counts(normalize=NORMALIZE))
df_viz_pred_counts = df_viz_pred_counts.reset_index().rename(columns={"label_text_pred": "label_count"}).rename(columns={"level_1": "label_text_pred"})

# harmonise df format to fix random bug
if len(df_viz_pred_counts) == 3:
    meta_data_col = df_viz_pred_counts[META_DATA].tolist() * 3
    label_col = df_viz_pred_counts.columns[1:4].tolist()
    label_col = label_col * 3
    label_values_col = [df_viz_pred_counts[col].tolist() for col in np.unique(label_col)]
    label_values_col = [item for sublist in label_values_col for item in sublist]
    df_viz_pred_counts = pd.DataFrame({META_DATA: meta_data_col, "label_text_pred": label_col, "label_count": label_values_col})

# enforce ordering of columns to get custom column order
if META_DATA == "parfam_text":
    parfam_paper_order = ["ECO", "ETH", "AGR", "LEF", "CHR", "LIB", "SOC", "CON", "NAT"]
    parfam_lr_order = ["ECO", "LEF", "SOC", "LIB", "CHR", "CON", "NAT", "AGR", "ETH"]
    df_viz_pred_counts[META_DATA] = pd.Categorical(df_viz_pred_counts[META_DATA], parfam_lr_order)
    df_viz_pred_counts = df_viz_pred_counts.sort_values([META_DATA, "label_text_pred"])

df_viz_pred_counts = df_viz_pred_counts[~df_viz_pred_counts[META_DATA].isna()]  # remove parfam (SIP) which is not in paper / categorical above


#try:
x_axis = []
data_dic_pred = {"immigration_supportive": [], "immigration_sceptical": [], "immigration_neutral": []}
for group_key, group_df_viz_pred in df_viz_pred_counts.groupby(by=META_DATA):
    x_axis.append(group_key)
    data_dic_pred["immigration_supportive"].append(group_df_viz_pred.iloc[2]["label_count"])
    data_dic_pred["immigration_sceptical"].append(group_df_viz_pred.iloc[1]["label_count"])
    data_dic_pred["immigration_neutral"].append(group_df_viz_pred.iloc[0]["label_count"])
"""except:
    print("! Exception triggered")
    x_axis = []
    data_dic_pred = {"immigration_supportive": [], "immigration_sceptical": [], "immigration_neutral": []}
    for group_key, group_df_viz_pred in df_viz_pred_counts.groupby(by=META_DATA):
        x_axis.append(group_key)
        data_dic_pred["immigration_supportive"].append(group_df_viz_pred["immigration_supportive"].iloc[0])
        data_dic_pred["immigration_sceptical"].append(group_df_viz_pred["immigration_sceptical"].iloc[0])
        data_dic_pred["immigration_neutral"].append(group_df_viz_pred["immigration_neutral"].iloc[0])
"""

fig_pred = go.Figure()
fig_pred.add_bar(x=x_axis, y=data_dic_pred["immigration_supportive"], name="immigration_supportive")
fig_pred.add_bar(x=x_axis, y=data_dic_pred["immigration_neutral"], name="immigration_neutral")
fig_pred.add_bar(x=x_axis, y=data_dic_pred["immigration_sceptical"], name="immigration_sceptical")
fig_pred.update_layout(barmode="relative", title=f"Predicted - trained on: {language_str} - Method: {METHOD}-{VECTORIZER}")
#fig_pred.show()



### 2. true distribution
# no-topic prediction irrelevant for the substantive task
df_viz_true = df[df.label_text != "no_topic"]
# prediction only for languages not in training data
df_viz_true = df_viz_true[~df_viz_true.language_iso.isin(languages_lst)]

df_viz_true_counts = df_viz_true.groupby(by=META_DATA, as_index=True, group_keys=True).apply(lambda x: x.label_text.value_counts(normalize=NORMALIZE))
df_viz_true_counts = df_viz_true_counts.reset_index().rename(columns={"label_text": "label_count"}).rename(columns={"level_1": "label_text"})

# adjust df in case of weird bug
if len(df_viz_true_counts) == 3:
    meta_data_col = df_viz_true_counts[META_DATA].tolist() * 3
    label_col = df_viz_true_counts.columns[1:4].tolist()
    label_col = label_col * 3
    label_values_col = [df_viz_true_counts[col].tolist() for col in np.unique(label_col)]
    label_values_col = [item for sublist in label_values_col for item in sublist]
    df_viz_true_counts = pd.DataFrame({META_DATA: meta_data_col, "label_text": label_col, "label_count": label_values_col})


# enforce ordering of columns to get custom column order
if META_DATA == "parfam_text":
    parfam_paper_order = ["ECO", "ETH", "AGR", "LEF", "CHR", "LIB", "SOC", "CON", "NAT"]
    parfam_lr_order = ["ECO", "LEF", "SOC", "LIB", "CHR", "CON", "NAT", "AGR", "ETH"]
    df_viz_true_counts[META_DATA] = pd.Categorical(df_viz_true_counts[META_DATA], parfam_lr_order)
    df_viz_true_counts = df_viz_true_counts.sort_values([META_DATA, "label_text"])

df_viz_true_counts = df_viz_true_counts[~df_viz_true_counts[META_DATA].isna()]  # remove parfam (SIP) which is not in paper / categorical above


#try:
x_axis = []
data_dic_true = {"immigration_supportive": [], "immigration_sceptical": [], "immigration_neutral": []}
for group_key, group_df_viz_true in df_viz_true_counts.groupby(by=META_DATA):
    x_axis.append(group_key)
    data_dic_true["immigration_supportive"].append(group_df_viz_true.iloc[2]["label_count"])
    data_dic_true["immigration_sceptical"].append(group_df_viz_true.iloc[1]["label_count"])
    data_dic_true["immigration_neutral"].append(group_df_viz_true.iloc[0]["label_count"])
"""except:
    print("! Exception triggered")
    x_axis = []
    data_dic_true = {"immigration_supportive": [], "immigration_sceptical": [], "immigration_neutral": []}
    for group_key, group_df_viz_true in df_viz_true_counts.groupby(by=META_DATA):
        x_axis.append(group_key)
        data_dic_true["immigration_supportive"].append(group_df_viz_true["immigration_supportive"].iloc[0])
        data_dic_true["immigration_sceptical"].append(group_df_viz_true["immigration_sceptical"].iloc[0])
        data_dic_true["immigration_neutral"].append(group_df_viz_true["immigration_neutral"].iloc[0])
"""
fig_true = go.Figure()
fig_true.add_bar(x=x_axis, y=data_dic_true["immigration_supportive"], name="immigration_supportive")
fig_true.add_bar(x=x_axis, y=data_dic_true["immigration_neutral"], name="immigration_neutral")
fig_true.add_bar(x=x_axis, y=data_dic_true["immigration_sceptical"], name="immigration_sceptical")
fig_true.update_layout(barmode="relative", title="True", showlegend=True)
#fig_true.show()



## try making subplots
from plotly.subplots import make_subplots  # https://plotly.com/python/subplots/
subplot_titles = ["Ground truth from crowd workers", f"Predicted - trained on: {language_str} - Method: {METHOD}-{VECTORIZER}"]
fig_subplot = make_subplots(rows=1, cols=2, #start_cell="top-left", horizontal_spacing=0.1, vertical_spacing=0.2,
                    subplot_titles=subplot_titles,
                    x_title=META_DATA, y_title="Proportion of stances")

fig_subplot.add_traces(fig_true["data"], rows=1, cols=1)
fig_subplot.add_traces(fig_pred["data"], rows=1, cols=2)
#fig_subplot.append_trace(fig_true, 1, 1)
#fig_subplot.append_trace(fig_pred, 1, 2)
#fig_subplot.add_traces(fig_true["data"], rows=[1, 1, 1], cols=[1, 1, 1])
#fig_subplot.add_trace(fig_pred, row=1, col=2)
fig_subplot.update_layout(barmode="relative", title=f"Comparison of true and predicted data distribution for stance towards {TASK} with method: {METHOD}-{VECTORIZER}",
                          title_x=0.5)

fig_subplot.show(renderer="browser")


### calculating correlation

from scipy.stats import pearsonr

print("\n")
for label in df_viz_true_counts.label_text.unique():
    pearson_cor = pearsonr(df_viz_pred_counts[df_viz_pred_counts.label_text_pred == label].label_count, df_viz_true_counts[df_viz_true_counts.label_text == label].label_count)
    print(f"Person R for {label}: ", pearson_cor)

pearson_cor = pearsonr(df_viz_pred_counts.label_count, df_viz_true_counts.label_count)
print("Pearson R overall: ", pearson_cor)


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
               'f1_micro': f1_micro,
               #'accuracy_balanced': acc_balanced,
               'accuracy_not_b': acc_not_balanced,
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

# solve label issue in some dfs
df["label"] = pd.factorize(df["label_text"], sort=True)[0]

# ! computing labels for all 4 classes here including no-topic (on 3 not possible, because different length)
metrics_all = compute_metrics_standard(label_gold=df.label, label_pred=df.label_pred)
#metrics_stances = compute_metrics_standard(label_gold=df_viz_true.label, label_pred=df_viz_pred.label_pred)
#print("\n", metrics_all)




