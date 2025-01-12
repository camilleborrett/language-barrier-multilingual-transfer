
# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

# main args
parser.add_argument('-lang', '--languages', type=str, #nargs='+',
                    help='List of languages to iterate over. one string separated with separator and split in code.')
parser.add_argument('-samp_lang', '--max_sample_lang', type=int, #nargs='+',
                    help='Sample')
parser.add_argument('-max_e', '--max_epochs', type=int, #nargs='+',
                    help='number of epochs')
parser.add_argument('-date', '--study_date', type=str,
                    help='Date')
parser.add_argument('-t', '--task', type=str,
                    help='task about integration or immigration?')
parser.add_argument('-meth', '--method', type=str,
                    help='NLI or standard dl?')
parser.add_argument('-v', '--vectorizer', type=str,
                    help='en or multi?')
parser.add_argument('-hypo', '--hypothesis', type=str,
                    help='which hypothesis?')
parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
parser.add_argument('-max_l', '--max_length', type=int, #nargs='+',
                    help='max n tokens')
parser.add_argument('-size', '--model_size', type=str,
                    help='base or large')


## choose arguments depending on execution in terminal or in script for testing
# ! does not work reliably in different environments
import sys
if len(sys.argv) > 1:
  print("Arguments passed via the terminal:")
  args = parser.parse_args()
  # To show the results of the given option to screen.
  print("")
  for key, value in parser.parse_args()._get_kwargs():
    print(value, "  ", key)
else:
  # parse args if not in terminal, but in script
  args = parser.parse_args(["--languages", "en-de", "--max_epochs", "2", "--task", "immigration", "--vectorizer", "multi",
                            "--method", "dl_embed", "--hypothesis", "long", "--nmt_model", "m2m_100_1.2B", "--max_length", "256",
                            "--max_sample_lang", "100", "--study_date", "221111"])

LANGUAGE_LST = args.languages.split("-")

MAX_SAMPLE_LANG = args.max_sample_lang
DATE = args.study_date
#MAX_EPOCHS = args.max_epochs
TASK = args.task
METHOD = args.method
VECTORIZER = args.vectorizer
HYPOTHESIS = args.hypothesis
MT_MODEL = args.nmt_model
#MODEL_MAX_LENGTH = args.max_length
MODEL_SIZE = args.model_size

SAMPLE_NO_TOPIC = 50_000  #100_000
#SAMPLE_DF_TEST = 1_000

## set main arguments
SEED_GLOBAL = 42
DATASET = "pimpo"

TRAINING_DIRECTORY = f"results/{DATASET}"


if (VECTORIZER == "en") and (METHOD == "dl_embed"):
    MODEL_NAME = "logistic"  #"transf_embed_en"
elif (VECTORIZER == "multi") and (METHOD == "dl_embed"):
    MODEL_NAME = "logistic"  #"transf_embed_multi"
else:
    raise Exception(f"VECTORIZER {VECTORIZER} or METHOD {METHOD} not implemented")




## load relevant packages
import pandas as pd
import numpy as np
import os
import torch
import datasets
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments

## Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)
from helpers import compute_metrics_standard, clean_memory, compute_metrics_nli_binary
#from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer, format_nli_trainset, format_nli_testset


##### load dataset
df = pd.read_csv(f"./data-clean/df_pimpo_samp_trans_{MT_MODEL}_embed_tfidf.zip", engine='python')


### inspect data
## inspect label distributions
# language
inspection_lang_dic = {}
for lang in df.language_iso.unique():
    inspection_lang_dic.update({lang: df[df.language_iso == lang].label_text.value_counts()})
df_inspection_lang = pd.DataFrame(inspection_lang_dic)
# party family
inspection_parfam_dic = {}
for parfam in df.parfam_text.unique():
    inspection_parfam_dic.update({parfam: df[df.parfam_text == parfam].label_text.value_counts()})
df_inspection_parfam = pd.DataFrame(inspection_parfam_dic)


### select training data

# choose bigger text window to improve performance and imitate annotation input
if VECTORIZER == "multi":
    if METHOD == "standard_dl":
        df["text_prepared"] = df["text_preceding"].fillna('') + " " + df["text_original"] + " " + df["text_following"].fillna('')
    elif METHOD == "nli":
        df["text_prepared"] = df["text_preceding"].fillna('') + '. The quote: "' + df["text_original"] + '". ' + df["text_following"].fillna('')
    elif METHOD == "dl_embed":
        df["text_prepared"] = df["text_concat_embed_multi"]
elif VECTORIZER == "en":
    if METHOD == "standard_dl":
        df["text_prepared"] = df["text_preceding_trans"].fillna('') + ' ' + df["text_original_trans"] + ' ' + df["text_following_trans"].fillna('')
    elif METHOD == "nli":
        df["text_prepared"] = df["text_preceding_trans"].fillna('') + '. The quote: "' + df["text_original_trans"] + '". ' + df["text_following_trans"].fillna('')
    elif METHOD == "dl_embed":
        df["text_prepared"] = df["text_trans_concat_embed_en"]
else:
    raise Exception(f"Vectorizer {VECTORIZER} not implemented.")



# select task
if TASK == "integration":
    task_label_text = ["integration_supportive", "integration_sceptical", "integration_neutral", "no_topic"]
elif TASK == "immigration":
    task_label_text = ["immigration_supportive", "immigration_sceptical", "immigration_neutral", "no_topic"]
#df_cl = df[df.label_text.isin(immigration_label_text)]
# replace labels for other task with "no_topic"
df_cl = df.copy(deep=True)
df_cl["label_text"] = [label if label in task_label_text else "no_topic" for label in df.label_text]
print(df_cl["label_text"].value_counts())

# adapt numeric label
df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]

## remove x% no_topic for faster testing
df_cl = df_cl.groupby(by="label_text", as_index=False, group_keys=False).apply(lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_GLOBAL) if x.label_text.iloc[0] == "no_topic" else x)
df_cl["label_text"].value_counts()



## select training data
# via language
#df_train = df_cl[df_cl.language_iso.isin(["en"])]
df_train = df_cl[df_cl.language_iso.isin(LANGUAGE_LST)]

# take sample for all topical labels - should not be more than SAMPLE (e.g. 500) per language to simulate realworld situation and prevent that adding some languages adds much more data than adding other languages - theoretically each coder can code n SAMPLE data
task_label_text_wo_notopic = task_label_text[:3]
df_train_samp1 = df_train.groupby(by="language_iso", as_index=False, group_keys=False).apply(lambda x: x[x.label_text.isin(task_label_text_wo_notopic)].sample(n=min(len(x[x.label_text.isin(task_label_text_wo_notopic)]), MAX_SAMPLE_LANG), random_state=SEED_GLOBAL))
df_train_samp2 = df_train.groupby(by="language_iso", as_index=False, group_keys=False).apply(lambda x: x[x.label_text == "no_topic"].sample(n=min(len(x[x.label_text == "no_topic"]), MAX_SAMPLE_LANG), random_state=SEED_GLOBAL))
df_train = pd.concat([df_train_samp1, df_train_samp2])
# could have also used len(df_train_samp1) for sampling df_train_samp2 instead of max_samp. Then could have avoided three lines below and possible different numbers in sampling across languages.

# for df_train reduce n no_topic data to same length as all topic data combined
df_train_topic = df_train[df_train.label_text != "no_topic"]
df_train_no_topic = df_train[df_train.label_text == "no_topic"].sample(n=min(len(df_train_topic), len(df_train[df_train.label_text == "no_topic"])), random_state=SEED_GLOBAL)
df_train = pd.concat([df_train_topic, df_train_no_topic])

print("\n", df_train.label_text.value_counts())
print(df_train.language_iso.value_counts(), "\n")

# avoid random overlap between topic and no-topic ?
# ! not sure if this adds value - maybe I even want overlap to teach it to look only at the middle sentence? Or need to check overlap by 3 text columns ?
#print("Is there overlapping text between train and test?", df_train_no_topic[df_train_no_topic.text_original.isin(df_train_topic.text_prepared)])

## create df test
# just to get accuracy figure as benchmark, less relevant for substantive use-case
df_test = df_cl[~df_cl.index.isin(df_train.index)]
assert len(df_train) + len(df_test) == len(df_cl)

# ! sample for faster testing
#df_test = df_test.sample(n=min(SAMPLE_DF_TEST, len(df_test)), random_state=SEED_GLOBAL)



if METHOD == "standard_dl":
    df_train_format = df_train
    df_test_format = df_test
#elif METHOD == "nli":
    #df_train_format = format_nli_trainset(df_train=df_train, hypo_label_dic=hypo_label_dic, random_seed=42)
    #df_test_format = format_nli_testset(df_test=df_test, hypo_label_dic=hypo_label_dic)
elif METHOD == "dl_embed":
    df_train_format = df_train
    df_test_format = df_test



##### train classifier

## ! do hp-search in separate script. should not take too long
## hyperparameters for final tests
# selective load one decent set of hps for testing
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results/manifesto-8/optuna_study_SVM_tfidf_01000samp_20221006.pkl")

# select best hp based on hp-search
n_sample_string = MAX_SAMPLE_LANG
#n_sample_string = 300
while len(str(n_sample_string)) <= 4:
    n_sample_string = "0" + str(n_sample_string)

import joblib
hp_study_dic = joblib.load(f"./{TRAINING_DIRECTORY}/optuna/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample_string}samp_{DATASET}_{'-'.join(LANGUAGE_LST)}_{MT_MODEL}_{DATE}.pkl")
hp_study_dic = next(iter(hp_study_dic.values()))  # unnest dic

hyperparams = hp_study_dic['optuna_study'].best_trial.user_attrs["hyperparameters_all"]


### text pre-processing
# separate hyperparams for vectorizer and classifier.
hyperparams_vectorizer = {key: value for key, value in hyperparams.items() if key in ["ngram_range", "max_df", "min_df", "analyzer"]}
hyperparams_clf = {key: value for key, value in hyperparams.items() if key not in ["ngram_range", "max_df", "min_df", "analyzer"]}
# in case I want to add tfidf later
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_sklearn = TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", use_idf=True, smooth_idf=True, **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02, token_pattern="(?u)\b\w\w+\b"


# choose correct pre-processed text column here
import ast
if VECTORIZER == "tfidf":
    # fit vectorizer on entire dataset - theoretically leads to some leakage on feature distribution in TFIDF (but is very fast, could be done for each test. And seems to be common practice) - OOV is actually relevant disadvantage of classical ML  #https://github.com/vanatteveldt/ecosent/blob/master/src/data-processing/19_svm_gridsearch.py
    vectorizer_sklearn.fit(pd.concat([df_train_format.text_trans_concat_tfidf, df_test_format.text_trans_concat_tfidf]))
    X_train = vectorizer_sklearn.transform(df_train_format.text_trans_concat_tfidf)
    X_test = vectorizer_sklearn.transform(df_test_format.text_trans_concat_tfidf)
elif "en" == VECTORIZER:
    X_train = np.array([ast.literal_eval(lst) for lst in df_train_format.text_trans_concat_embed_en.astype('object')])
    X_test = np.array([ast.literal_eval(lst) for lst in df_test_format.text_trans_concat_embed_en.astype('object')])
elif "multi" == VECTORIZER:
    X_train = np.array([ast.literal_eval(lst) for lst in df_train_format.text_concat_embed_multi.astype('object')])
    X_test = np.array([ast.literal_eval(lst) for lst in df_test_format.text_concat_embed_multi.astype('object')])

y_train = df_train_format.label
y_test = df_test_format.label

## initialise and train classifier
from sklearn import svm, linear_model
if MODEL_NAME == "SVM":
    clf = svm.SVC(**hyperparams_clf)
elif MODEL_NAME == "logistic":
    clf = linear_model.LogisticRegression(**hyperparams_clf)
clf.fit(X_train, y_train)



### Evaluate
# test on test set
label_gold = y_test
label_pred = clf.predict(X_test)

### metrics
from helpers import compute_metrics_classical_ml
#results_test = trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
results_test = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df_cl.label_text.unique()))

print(results_test)

# do prediction on entire corpus? No.
# could do because annotations also contribute to estimate of distribution
#dataset["all"] = datasets.concatenate_datasets([dataset["test"], dataset["train"]])
#results_corpus = trainer.evaluate(eval_dataset=datasets.concatenate_datasets([dataset["train"], dataset["test"]]))  # eval_dataset=encoded_dataset["test"]
# with NLI, cannot run inference also on train set, because augmented train set can have different length than original train-set


#### prepare data for redoing figure from paper

assert (df_test["label"] == results_test["eval_label_gold_raw"]).all
df_test["label_pred"] = results_test["eval_label_predicted_raw"]

df_train["label_pred"] = [np.nan] * len(df_train["label"])

df_cl_concat = pd.concat([df_train, df_test])

# add label text for predictions
label_text_map = {}
for i, row in df_cl_concat[~df_cl_concat.label_text.duplicated(keep='first')].iterrows():
    label_text_map.update({row["label"]: row["label_text"]})
df_cl_concat["label_text_pred"] = df_cl_concat["label_pred"].map(label_text_map)


## trim df to save storage
df_cl_concat = df_cl_concat[['label', 'label_text', 'country_iso', 'language_iso', 'doc_id',
                           'text_original', 'text_original_trans', 'text_preceding_trans', 'text_following_trans',
                           # 'text_preceding', 'text_following', 'selection', 'certainty_selection', 'topic', 'certainty_topic', 'direction', 'certainty_direction',
                           'rn', 'cmp_code', 'partyname', 'partyabbrev',
                           'parfam', 'parfam_text', 'date', #'language_iso_fasttext', 'language_iso_trans',
                           #'text_concat', 'text_concat_embed_multi', 'text_trans_concat',
                           #'text_trans_concat_embed_en', 'text_trans_concat_tfidf', 'text_prepared',
                           'label_pred', 'label_text_pred']]

## save data
langs_concat = "_".join(LANGUAGE_LST)
#df_cl_concat.to_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.csv", index=False)
df_cl_concat.to_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.zip",
                    compression={"method": "zip", "archive_name": f"df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.csv"}, index=False)



print("Script done.")





