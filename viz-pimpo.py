

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers_pimpo
import importlib  # in case of manual updates in .py file
importlib.reload(helpers_pimpo)

from helpers_pimpo import load_data, calculate_distribution, compute_correlation, compute_metrics_standard


### create overview df

LANGUAGES = ["en", "de", "sv", "fr"]  # [["en"], ["en", "de"], ["en", "de", "sv", "fr"]]
DATE = "221111"
TASK = "immigration"  # "immigration", "integration
METHOD = "nli"  # standard_dl nli
VECTORIZER = "en" # ["en", "multi"]
HYPOTHESIS = "long"
MAX_SAMPLE_LANG = "500"
META_DATA = "decade"  #["parfam_text", "country_iso", "language_iso", "decade"]
MODEL_SIZE = "large"  #["base", "large"]
NORMALIZE = True
EXCLUDE_NO_TOPIC = True


df_cl = load_data(languages=LANGUAGES, task=TASK, method=METHOD, model_size=MODEL_SIZE, hypothesis=HYPOTHESIS, vectorizer=VECTORIZER, max_sample_lang=MAX_SAMPLE_LANG, date=DATE)
df_viz_pred_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA)
df_viz_true_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA)
corr_dic, df_merge = compute_correlation(df_viz_true_counts=df_viz_true_counts, df_viz_pred_counts=df_viz_pred_counts, meta_data=META_DATA, df=df_cl)
metrics_dic = compute_metrics_standard(label_gold=df_cl.label, label_pred=df_cl.label_pred)







##### visualisations
def create_figure(df_func=None, df_counts_func=None, label_count_col="label_count_pred"):
    x_axis = []
    #data_dic = {label: [] for label in df_func.label_text.unique()}
    data_dic = {label: [] for label in df_counts_func.label_text.unique()}
    for group_key, group_df_viz in df_counts_func.groupby(by=META_DATA):
        x_axis.append(group_key)
        # append label count for each label to data_dic with respective label key
        for key_label_text in data_dic:
            data_dic[key_label_text].append(group_df_viz[group_df_viz["label_text"] == key_label_text][label_count_col].iloc[0])

    fig = go.Figure()
    for key_label_text in data_dic:
        fig.add_bar(x=x_axis, y=data_dic[key_label_text], name=key_label_text)

    #if df_label_col == "label_text_pred":
    #    fig.update_layout(barmode="relative", title=f"Predicted - trained on: {language_str} - Method: {METHOD}-{VECTORIZER}")
    #else:
    fig.update_layout(barmode="relative", title=f"True")

    return fig


language_str = "_".join(LANGUAGES)

fig_pred = create_figure(df_func=df_cl, df_counts_func=df_merge, label_count_col="label_count_pred")  # df_label_col="label_text_pred"
fig_true = create_figure(df_func=df_cl, df_counts_func=df_merge, label_count_col="label_count_true")  # df_label_col="label_text"
fig_pred.update_layout(barmode="relative", title=f"Predicted - trained on: {language_str} - Method: {METHOD}-{VECTORIZER}")

## try making subplots
from plotly.subplots import make_subplots  # https://plotly.com/python/subplots/

subplot_titles = ["Ground truth from crowd workers", f"Predicted - trained on: {language_str} - Method: {METHOD}-{VECTORIZER}"]
fig_subplot = make_subplots(rows=1, cols=2,  # start_cell="top-left", horizontal_spacing=0.1, vertical_spacing=0.2,
                            subplot_titles=subplot_titles,
                            x_title=META_DATA, y_title="Proportion of stances")

fig_subplot.add_traces(fig_true["data"], rows=1, cols=1)
fig_subplot.add_traces(fig_pred["data"], rows=1, cols=2)
fig_subplot.update_layout(barmode="relative", title=f"Comparison of true and predicted data distribution for stance towards {TASK} with method: {METHOD}-{VECTORIZER}",
                          title_x=0.5)
fig_subplot.show(renderer="browser")




print("Run done.")

