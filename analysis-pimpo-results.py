

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np

# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers_pimpo
import importlib  # in case of manual updates in .py file
importlib.reload(helpers_pimpo)

from helpers_pimpo import load_data, calculate_distribution, compute_correlation, compute_metrics_standard


"""LANGUAGES = ["en"] # ["en"] ["en", "de"] ["en", "de", "sv", "fr"]
DATE = "221111"
TASK = "immigration"  # "immigration", "integration
METHOD = "nli"  # standard_dl nli
VECTORIZER = "en"
HYPOTHESIS = "long"
MAX_SAMPLE_LANG = "500"
MODEL_SIZE = "base"  # base, large

META_DATA = "parfam_text"  # "parfam_text", "country_iso", "language_iso", "decade"
NORMALIZE = True
EXCLUDE_NO_TOPIC = True
"""

## load data
#df_cl = load_data()

## Calculate predicted or true distribution by meta data
#df_viz_pred_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred")
#df_viz_true_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text")

## calculating correlation
#corr_dic = compute_correlation(df_viz_true_counts=df_viz_true_counts, df_viz_pred_counts=df_viz_pred_counts)


## calculating classical ml metrics
# ! computing labels for all 4 classes here including no-topic (on 3 not possible, because different length)
#metrics_dic = compute_metrics_standard(label_gold=df_cl.label, label_pred=df_cl.label_pred)


### create overview df

LANGUAGES_LST = [["en"], ["en", "de"], ["en", "de", "sv", "fr"]]
DATE = "221111"
TASK_LST = ["immigration"]  # "immigration", "integration
METHOD_LST = ["standard_dl", "nli"]  # standard_dl nli
VECTORIZER_LST = ["en", "multi"]
HYPOTHESIS = "long"
MAX_SAMPLE_LANG = "500"
META_DATA_LST = ["parfam_text", "country_iso", "language_iso", "decade"]
MODEL_SIZE_LST = ["base", "large"]
NORMALIZE = True
EXCLUDE_NO_TOPIC = False

# prediction only for languages not in training data
data_test = []

#for TASK in TASK_LST:
TASK = "immigration"
for METHOD in METHOD_LST:
    for MODEL_SIZE in MODEL_SIZE_LST:
        for VECTORIZER in VECTORIZER_LST:
            for LANGUAGES in LANGUAGES_LST:
                # large multilingual does not exist
                if not ((MODEL_SIZE == "large") and (VECTORIZER == "multi")):
                    # load data here to save compute
                    df_cl = load_data(languages=LANGUAGES, task=TASK, method=METHOD, model_size=MODEL_SIZE, hypothesis=HYPOTHESIS, vectorizer=VECTORIZER, max_sample_lang=MAX_SAMPLE_LANG, date=DATE)
                    for META_DATA in META_DATA_LST:
                        df_viz_pred_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA)
                        df_viz_true_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA)
                        corr_dic, df_merge = compute_correlation(df_viz_true_counts=df_viz_true_counts, df_viz_pred_counts=df_viz_pred_counts, meta_data=META_DATA, df=df_cl)
                        metrics_dic = compute_metrics_standard(label_gold=df_cl.label, label_pred=df_cl.label_pred)
                        data_test.append({"meta_data": META_DATA, "method": METHOD, "vectorizer": VECTORIZER, "size": MODEL_SIZE, "languages": "-".join(LANGUAGES), **corr_dic, **metrics_dic})

df_results = pd.DataFrame(data_test)
#df_results = df_results.sort_values(by=["corr_labels_all"], ascending=False)
df_results = df_results.sort_values(by=["meta_data", "corr_labels_avg"], ascending=False)  # "method", "vectorizer", "languages"
#df_results = df_results.round(2)  # do not round here yet for more precide correlation / p-value calculation
df_results = df_results[['meta_data', 'method', 'vectorizer', 'size', 'languages',
       'corr_labels_avg', 'f1_macro',
       'accuracy', 'corr_labels_all', 'p-value-labels_all', 'corr_immigration_neutral',
       'p-value-immigration_neutral', 'corr_immigration_sceptical',
       'p-value-immigration_sceptical', 'corr_immigration_supportive',
       'p-value-immigration_supportive', 'precision_macro', 'recall_macro', 'precision_micro',
       'recall_micro']]

# !? issue with standard_dl, en, base, if only trained on en


### correlate performance with different methods
from scipy.stats import spearmanr
#pearsonr(df_results.method, df_results.corr_labels_avg)
df_results_corr = df_results.copy(deep=True)
df_results_corr["method"] = pd.Categorical(df_results_corr["method"], ["standard_dl", "nli"])
df_results_corr["size"] = pd.Categorical(df_results_corr["size"], ["base", "large"])
df_results_corr["vectorizer"] = pd.Categorical(df_results_corr["vectorizer"], ["en", "multi"])
df_results_corr["languages"] = pd.Categorical(df_results_corr["languages"], ["en", "en-de", "en-de-sv-fr"])
df_results_corr["method_factor"] = df_results_corr["method"].factorize(sort=True)[0]
df_results_corr["size_factor"] = df_results_corr["size"].factorize(sort=True)[0]
df_results_corr["vectorizer_factor"] = df_results_corr["vectorizer"].factorize(sort=True)[0]
df_results_corr["languages_factor"] = df_results_corr["languages"].factorize(sort=True)[0]

#corr_spearman = spearmanr(df_results_corr[["method_factor", "size_factor", "vectorizer_factor", "languages_factor", "corr_labels_avg"]])  # , df_results_corr.corr_labels_avg
df_results_corr = df_results_corr[["method_factor", "size_factor", "vectorizer_factor", "languages_factor", "corr_labels_avg", "f1_macro", "accuracy"]]
df_corr_spearman = df_results_corr.corr(method="spearman")
# add p-value / significance
#rho = df.corr()
df_corr_pval = df_corr_spearman.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*df_corr_spearman.shape)
df_corr_pval = df_corr_pval.applymap(lambda x: ''.join(['*' for t in [0.001, 0.01, 0.05] if x<=t]))
df_results_corr = df_corr_spearman.round(2).astype(str) + df_corr_pval
df_results = df_results.round(2)

print("Run Done.")


