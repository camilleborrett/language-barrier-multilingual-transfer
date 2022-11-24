

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
import helpers_a2
import importlib  # in case of manual updates in .py file
importlib.reload(helpers_a2)

from helpers_a2 import load_data, calculate_distribution, compute_correlation, compute_metrics_standard


"""LANGUAGES = ["en", "de", "sv", "fr"]  # [["en"], ["en", "de"], ["en", "de", "sv", "fr"]]
DATE = "221111"
TASK = "immigration"  # "immigration", "integration
METHOD = "nli"  # standard_dl nli
VECTORIZER = "multi"  # ["en", "multi"]
HYPOTHESIS = "long"
MAX_SAMPLE_LANG = "500"
META_DATA = "language_iso"  #["parfam_text", "country_iso", "language_iso", "decade"]
MODEL_SIZE = "base"   #["base", "large"]
NORMALIZE = True
EXCLUDE_NO_TOPIC = True"""

## load data
# ! decide whether to exclude also test texts from train langs (for cross-ling transfer tests) or include them for accurate population simulation
# going for inclusion of test texts from all languages, also test form lang train. reflects population better and scenarios across decades etc. have more data
#df_cl = load_data(languages=LANGUAGES, task=TASK, method=METHOD, model_size=MODEL_SIZE, hypothesis=HYPOTHESIS, vectorizer=VECTORIZER, max_sample_lang=MAX_SAMPLE_LANG, date=DATE)


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
METHOD_LST = ["standard_dl", "nli", "dl_embed"]  # standard_dl nli dl_embed
VECTORIZER_LST = ["en", "multi"]
HYPOTHESIS = "long"
MAX_SAMPLE_LANG = "500"
META_DATA_LST = ["parfam_text", "country_iso", "language_iso", "decade"]
MODEL_SIZE_LST = ["base", "large", "classical"]  # classical
NORMALIZE = True
EXCLUDE_NO_TOPIC = True
META_METRICS_ANALYSIS = False
# ? why does including no-topic improve correlation for all (non-parfam) meta data by so much?  # ?! performances for other meta data get better when excluding no-topic via upstream:  df_cl[df_cl.label_text != "no_topic"]
# possibly because no-topic is so large that it overcrowds the other labels and makes the differences between them smaller. easier for classifier to predict large no-topic correctly? it's always largest

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
                    if not ((MODEL_SIZE == "classical") and (METHOD not in ["dl_embed"] )) and not ((MODEL_SIZE != "classical") and (METHOD in ["dl_embed"])):
                        # load data here to save compute
                        df_cl, df_train = load_data(languages=LANGUAGES, task=TASK, method=METHOD, model_size=MODEL_SIZE, hypothesis=HYPOTHESIS, vectorizer=VECTORIZER, max_sample_lang=MAX_SAMPLE_LANG, date=DATE)
                        #if EXCLUDE_NO_TOPIC: df_cl = df_cl[df_cl.label_text != "no_topic"]  # ! but no-topic can remain in label_text_pred !
                        for META_DATA in META_DATA_LST:
                            df_viz_pred_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
                            df_viz_true_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
                            corr_dic, df_merge = compute_correlation(df_viz_true_counts=df_viz_true_counts, df_viz_pred_counts=df_viz_pred_counts, exclude_no_topic=EXCLUDE_NO_TOPIC,meta_data=META_DATA, df=df_cl)
                            metrics_dic = compute_metrics_standard(label_gold=df_cl.label, label_pred=df_cl.label_pred)
                            # try and calculate classical metrics also by meta-data variables - see if this could explain performance difference. maybe NLI performs also most homogenously across meta-data, while others perform better on majority meta-data
                            if META_METRICS_ANALYSIS:
                                metrics_by_meta = df_cl.groupby(by=META_DATA, as_index=True, group_keys=True).apply(lambda x: pd.Series(compute_metrics_standard(label_gold=x.label, label_pred=x.label_pred)))
                                metrics_by_meta_mean = metrics_by_meta.mean().to_dict()
                                metrics_by_meta_std = metrics_by_meta.std().to_dict()
                                metrics_by_meta_mean = {key+"_meta_mean": value for key, value in metrics_by_meta_mean.items() if key in ["f1_macro", "accuracy"]}
                                metrics_by_meta_std = {key+"_meta_std": value for key, value in metrics_by_meta_std.items() if key in ["f1_macro", "accuracy"]}
                            else:
                                metrics_by_meta_mean, metrics_by_meta_std = {}, {}
                            data_test.append({"meta_data": META_DATA, "method": METHOD, "vectorizer": VECTORIZER, "size": MODEL_SIZE, "languages": "-".join(LANGUAGES), **corr_dic, **metrics_dic, **metrics_by_meta_mean, **metrics_by_meta_std})

## extrapolation analysis
# correlate training data distribution with population distribution - to test how (oracle) sample would perform
for TASK in TASK_LST:
    for LANGUAGES in LANGUAGES_LST:
        for META_DATA in META_DATA_LST:
            # ? should be OK to just load for any  method / size etc, because not using model predictions, only need data for language scenarios
            df_cl, df_train = load_data(languages=LANGUAGES, task=TASK, method=METHOD, model_size=MODEL_SIZE, hypothesis=HYPOTHESIS, vectorizer=VECTORIZER, max_sample_lang=MAX_SAMPLE_LANG, date=DATE)
            # add sample extrapolation test
            df_viz_true_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA)
            df_viz_train_counts = calculate_distribution(df_func=df_train, df_label_col="label_text", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA)
            corr_dic, df_merge = compute_correlation(df_viz_true_counts=df_viz_true_counts, df_viz_pred_counts=df_viz_train_counts, exclude_no_topic=EXCLUDE_NO_TOPIC, meta_data=META_DATA, df=df_cl)
            data_test.append({"meta_data": META_DATA, "method": "extrapolation-train", "vectorizer": None, "size": None, "languages": "-".join(LANGUAGES), **corr_dic})  # **metrics_dic, **metrics_by_meta_mean, **metrics_by_meta_std


# create df
df_results = pd.DataFrame(data_test)
# sort
df_results["method"] = pd.Categorical(df_results["method"], ["extrapolation-train", "dl_embed", "standard_dl", "nli"])
df_results = df_results.sort_values(by=["meta_data", "method", "corr_labels_avg"], ascending=False)  # "method", "vectorizer", "languages"
# select relevant columns
df_results = df_results[['meta_data', 'method', 'vectorizer', 'size', 'languages',
       'corr_labels_avg', "corr_labels_p_avg", 'f1_macro', 'accuracy',
       # meta-metrics analysis
       #"f1_macro_meta_mean", "accuracy_meta_mean", "f1_macro_meta_std", "accuracy_meta_std",
       'corr_labels_all', 'p-value-labels_all', 'corr_immigration_neutral',
       'p-value-immigration_neutral', 'corr_immigration_sceptical',
       'p-value-immigration_sceptical', 'corr_immigration_supportive',
       'p-value-immigration_supportive', 'precision_macro', 'recall_macro', 'precision_micro',
       'recall_micro']]


#### Analysis which factor improves performance the most
# correlate performance with different methods
from scipy.stats import spearmanr
#pearsonr(df_results.method, df_results.corr_labels_avg)
df_results_corr = df_results.copy(deep=True)
df_results_corr["algorithm"] = pd.Categorical(df_results_corr["method"], ["extrapolation-train", "dl_embed", "standard_dl", "nli"])
df_results_corr["size"] = pd.Categorical(df_results_corr["size"], ["classical", "base", "large"])
df_results_corr["representation"] = pd.Categorical(df_results_corr["vectorizer"], ["en", "multi"])
df_results_corr["languages"] = pd.Categorical(df_results_corr["languages"], ["en", "en-de", "en-de-sv-fr"])
df_results_corr["algorithm_factor"] = df_results_corr["algorithm"].factorize(sort=True)[0]
df_results_corr["size_factor"] = df_results_corr["size"].factorize(sort=True)[0]
df_results_corr["representation_factor"] = df_results_corr["representation"].factorize(sort=True)[0]
df_results_corr["languages_factor"] = df_results_corr["languages"].factorize(sort=True)[0]

#corr_spearman = spearmanr(df_results_corr[["method_factor", "size_factor", "vectorizer_factor", "languages_factor", "corr_labels_avg"]])  # , df_results_corr.corr_labels_avg
df_results_corr = df_results_corr[["algorithm_factor", "size_factor", "representation_factor", "languages_factor", "corr_labels_avg", "f1_macro", "accuracy"]]
df_corr_spearman = df_results_corr.corr(method="spearman")
# add p-value / significance
df_corr_pval = df_corr_spearman.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*df_corr_spearman.shape)
df_corr_pval = df_corr_pval.applymap(lambda x: ''.join(['*' for t in [0.001, 0.01, 0.05] if x <= t]))
df_results_corr = df_corr_spearman.round(2).astype(str) + df_corr_pval


# rename some values and columns
df_results = df_results.rename(columns={"languages": "training languages", "f1_macro": "F1 macro", "method": "algorithm",
                                        "vectorizer": "language representation", "corr_labels_avg": "average correlation", "corr_labels_p_avg": "average p-value",
                                        "meta_data": "meta data", "size": "algorithm size", "decade": "time"})

# round
cols_round = ['average correlation', 'F1 macro', 'accuracy',
              # meta-metrics analysis
              #'f1_macro_meta_mean', 'accuracy_meta_mean', 'f1_macro_meta_std', 'accuracy_meta_std',
              'corr_labels_all', 'corr_immigration_neutral', 'corr_immigration_sceptical', 'corr_immigration_supportive',
              'precision_macro', 'recall_macro', 'precision_micro', 'recall_micro']
df_results[cols_round] = df_results[cols_round].round(2)
cols_round = ['average p-value', 'p-value-labels_all', 'p-value-immigration_neutral', 'p-value-immigration_sceptical', 'p-value-immigration_supportive']
df_results[cols_round] = df_results[cols_round].round(3)


algorithm_map = {
    "nli": "NLI-Transformer", "standard_dl": "Transformer", "dl_embed": "Sent.-Transformer", "extrapolation-train": "extrapolation-train"
}
df_results["algorithm"] = df_results["algorithm"].map(algorithm_map)
size_map = {
    "base": "base", "large": "large", "classical": "base"  # because classical logreg uses a base-sized transformer
}
df_results["algorithm size"] = df_results["algorithm size"].map(size_map)


## inspect data
df_results_parfam = df_results[df_results["meta data"] == "parfam_text"]
df_results_parfam = df_results_parfam.sort_values(by=["algorithm", "average correlation"], ascending=False)



## write to disk
df_results.to_excel("./results/pimpo/tables/df_results_pimpo.xlsx")
df_results_corr.to_excel("./results/pimpo/tables/df_results_pimpo_corr.xlsx")

print("Run Done.")




## test analysis of standard metric performance disaggregated by meta data
# hypothesis: NLI performs more evenly across meta-data splits (not only class splits). could explain better correlation
# ? weird, would expect f1_macro_meta to be proportionally higher and std lower, but the opposite is the case
if META_METRICS_ANALYSIS:
    mean_performance_dic = {}
    for metric in ["average correlation", "F1 macro", "accuracy", "f1_macro_meta_mean", "f1_macro_meta_std", "accuracy_meta_mean", "accuracy_meta_std"]:
        mean_performance = df_results.groupby(by="algorithm").apply(lambda x: x[x["algorithm size"] == "base"][metric].mean())
        mean_performance_dic.update({metric: mean_performance})
    df_mean_performance = pd.DataFrame(mean_performance_dic)








## deletable code for cleaning columns of oversized dfs
"""
LANGUAGES_LST = [["en"], ["en", "de"], ["en", "de", "sv", "fr"]]
DATE = "221111"
TASK_LST = ["immigration"]  # "immigration", "integration
METHOD_LST = ["standard_dl", "nli"]  # standard_dl nli dl_embed
VECTORIZER_LST = ["en", "multi"]
HYPOTHESIS = "long"
MAX_SAMPLE_LANG = "500"
META_DATA_LST = ["parfam_text", "country_iso", "language_iso", "decade"]
MODEL_SIZE_LST = ["base", "large"]  # classical
NORMALIZE = True
EXCLUDE_NO_TOPIC = True

# !! add AGR parfam again - still has some bug

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
                    df = pd.read_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{'_'.join(LANGUAGES)}_{DATE}.zip")
                    df_cl = df[['label', 'label_text', 'country_iso', 'language_iso', 'doc_id',
                                                 'text_original', 'text_original_trans', 'text_preceding_trans', 'text_following_trans',
                                                 # 'text_preceding', 'text_following', 'selection', 'certainty_selection', 'topic', 'certainty_topic', 'direction', 'certainty_direction',
                                                 'rn', 'cmp_code', 'partyname', 'partyabbrev',
                                                 'parfam', 'parfam_text', 'date',  # 'language_iso_fasttext', 'language_iso_trans',
                                                 # 'text_concat', 'text_concat_embed_multi', 'text_trans_concat',
                                                 # 'text_trans_concat_embed_en', 'text_trans_concat_tfidf', 'text_prepared',
                                                 'label_pred', 'label_text_pred']]

                    df_cl.to_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{'_'.join(LANGUAGES)}_{DATE}.zip",
                                        compression={"method": "zip",
                                                     "archive_name": f"df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{'_'.join(LANGUAGES)}_{DATE}.csv"},
                                        index=False)
"""
