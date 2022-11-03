

import pandas as pd
import plotly.graph_objects as go


languages_lst = ["en"]
languages_lst = ["en", "de"]
languages_lst = ["en", "de", "sv", "fr"]
DATE = "221103"
TASK = "immigration"  # "immigration", "integration

language_str = "_".join(languages_lst)

df = pd.read_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_500samp_{language_str}_{DATE}.csv")
df = df.rename(columns={"label_pred_text": "label_text_pred"})

NORMALIZE = True

### 1. predicted distribution
df_viz_pred = df[df.label_text_pred != "no_topic"]
df_viz_pred_counts = df_viz_pred.groupby(by="parfam_text", as_index=True, group_keys=True).apply(lambda x: x.label_text_pred.value_counts(normalize=NORMALIZE))
df_viz_pred_counts = df_viz_pred_counts.reset_index().rename(columns={"label_text_pred": "label_count"}).rename(columns={"level_1": "label_text_pred"})

# enforce ordering of columns to get custom column order
parfam_paper_order = ["ECO", "ETH", "AGR", "LEF", "CHR", "LIB", "SOC", "CON", "NAT"]
parfam_lr_order = ["ECO", "LEF", "SOC", "LIB", "CHR", "CON", "NAT", "AGR", "ETH"]
df_viz_pred_counts["parfam_text"] = pd.Categorical(df_viz_pred_counts["parfam_text"], parfam_lr_order)
df_viz_pred_counts = df_viz_pred_counts.sort_values(["parfam_text", "label_text_pred"])

df_viz_pred_counts = df_viz_pred_counts[~df_viz_pred_counts.parfam_text.isna()]  # remove parfam (SIP) which is not in paper / categorical above

x_axis = []
data_dic_pred = {"immigration_supportive": [], "immigration_sceptical": [], "immigration_neutral": []}
for group_key, group_df_viz_pred in df_viz_pred_counts.groupby(by="parfam_text"):
    x_axis.append(group_key)
    data_dic_pred["immigration_supportive"].append(group_df_viz_pred.iloc[2]["label_count"])
    data_dic_pred["immigration_sceptical"].append(group_df_viz_pred.iloc[1]["label_count"])
    data_dic_pred["immigration_neutral"].append(group_df_viz_pred.iloc[0]["label_count"])

fig_pred = go.Figure()
fig_pred.add_bar(x=x_axis, y=data_dic_pred["immigration_supportive"], name="immigration_supportive")
fig_pred.add_bar(x=x_axis, y=data_dic_pred["immigration_neutral"], name="immigration_neutral")
fig_pred.add_bar(x=x_axis, y=data_dic_pred["immigration_sceptical"], name="immigration_sceptical")
fig_pred.update_layout(barmode="relative", title=f"predicted - {language_str}")
fig_pred.show()



### 2. true distribution
df_viz_true = df[df.label_text != "no_topic"]
df_viz_true_counts = df_viz_true.groupby(by="parfam_text", as_index=True, group_keys=True).apply(lambda x: x.label_text.value_counts(normalize=NORMALIZE))
df_viz_true_counts = df_viz_true_counts.reset_index().rename(columns={"label_text": "label_count"}).rename(columns={"level_1": "label_text"})

# enforce ordering of columns to get custom column order
parfam_paper_order = ["ECO", "ETH", "AGR", "LEF", "CHR", "LIB", "SOC", "CON", "NAT"]
parfam_lr_order = ["ECO", "LEF", "SOC", "LIB", "CHR", "CON", "NAT", "AGR", "ETH"]
df_viz_true_counts["parfam_text"] = pd.Categorical(df_viz_true_counts["parfam_text"], parfam_lr_order)
df_viz_true_counts = df_viz_true_counts.sort_values(["parfam_text", "label_text"])

df_viz_true_counts = df_viz_true_counts[~df_viz_true_counts.parfam_text.isna()]  # remove parfam (SIP) which is not in paper / categorical above

x_axis = []
data_dic_true = {"immigration_supportive": [], "immigration_sceptical": [], "immigration_neutral": []}
for group_key, group_df_viz_true in df_viz_true_counts.groupby(by="parfam_text"):
    x_axis.append(group_key)
    data_dic_true["immigration_supportive"].append(group_df_viz_true.iloc[2]["label_count"])
    data_dic_true["immigration_sceptical"].append(group_df_viz_true.iloc[1]["label_count"])
    data_dic_true["immigration_neutral"].append(group_df_viz_true.iloc[0]["label_count"])

fig_true = go.Figure()
fig_true.add_bar(x=x_axis, y=data_dic_true["immigration_supportive"], name="immigration_supportive")
fig_true.add_bar(x=x_axis, y=data_dic_true["immigration_neutral"], name="immigration_neutral")
fig_true.add_bar(x=x_axis, y=data_dic_true["immigration_sceptical"], name="immigration_sceptical")
fig_true.update_layout(barmode="relative", title="True")
fig_true.show()

