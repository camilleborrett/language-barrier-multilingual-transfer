

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
  args = parser.parse_args(["--languages", "en-de", "--max_epochs", "2", "--task", "integration", "--vectorizer", "en",
                            "--method", "nli", "--hypothesis", "long", "--nmt_model", "m2m_100_418M", "--max_length", "256",
                            "--max_sample_lang", "50", "--study_date", "221111"])

LANGUAGE_LST = args.languages.split("-")

MAX_SAMPLE_LANG = args.max_sample_lang
DATE = args.study_date
#MAX_EPOCHS = args.max_epochs
TASK = args.task
METHOD = args.method
VECTORIZER = args.vectorizer
HYPOTHESIS = args.hypothesis
MT_MODEL = args.nmt_model
MODEL_MAX_LENGTH = args.max_length
MODEL_SIZE = args.model_size

SAMPLE_NO_TOPIC = 50_000  #100_000
#SAMPLE_DF_TEST = 1_000

## set main arguments
SEED_GLOBAL = 42
DATASET = "pimpo"

TRAINING_DIRECTORY = f"results/{DATASET}"


if (VECTORIZER == "en") and (METHOD == "transf_embed"):
    MODEL_NAME = "logistic"  #"transf_embed_en"
elif (VECTORIZER == "multi") and (METHOD == "transf_embed"):
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

# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)
from helpers import compute_metrics_standard, clean_memory, compute_metrics_nli_binary
#from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer, format_nli_trainset, format_nli_testset


##### load dataset
df = pd.read_csv(f"./data-clean/df_pimpo_samp_trans_{MT_MODEL}_embed_tfidf.zip")


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
# note that for this substantive use-case, it seems fine to predict also on training data,
# because it is about the substantive outcome of the approach, not out-of-sample accuracy

# choose bigger text window to improve performance and imitate annotation input
if VECTORIZER == "multi":
    df["text_prepared"] = df["text_original_trans_embed_multi"]
    #if METHOD == "standard_dl":
    #    df["text_prepared"] = df["text_preceding"].fillna('') + " " + df["text_original"] + " " + df["text_following"].fillna('')
    #elif METHOD == "nli":
    #    df["text_prepared"] = df["text_preceding"].fillna('') + '. The quote: "' + df["text_original"] + '". ' + df["text_following"].fillna('')
elif VECTORIZER == "en":
    df["text_prepared"] = df["text_original_trans_embed_en"]
    #if METHOD == "standard_dl":
    #    df["text_prepared"] = df["text_preceding_trans"].fillna('') + ' ' + df["text_original_trans"] + ' ' + df["text_following_trans"].fillna('')
    #elif METHOD == "nli":
    #    df["text_prepared"] = df["text_preceding_trans"].fillna('') + '. The quote: "' + df["text_original_trans"] + '". ' + df["text_following_trans"].fillna('')
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
df_cl["label_text"].value_counts()

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

print(df_train.label_text.value_counts())

# avoid random overlap between topic and no-topic ?
# ! not sure if this adds value - maybe I even want overlap to teach it to look only at the middle sentence? Or need to check overlap by 3 text columns ?
#print("Is there overlapping text between train and test?", df_train_no_topic[df_train_no_topic.text_original.isin(df_train_topic.text_prepared)])

## create df test
# just to get accuracy figure as benchmark, less relevant for substantive use-case
df_test = df_cl[~df_cl.index.isin(df_train.index)]
assert len(df_train) + len(df_test) == len(df_cl)

# sample for faster testing
#df_test = df_test.sample(n=min(SAMPLE_DF_TEST, len(df_test)), random_state=SEED_GLOBAL)



#if METHOD == "standard_dl":
#    df_train_format = df_train
#    df_test_format = df_test
#elif METHOD == "nli":
#    df_train_format = format_nli_trainset(df_train=df_train, hypo_label_dic=hypo_label_dic, random_seed=42)
#    df_test_format = format_nli_testset(df_test=df_test, hypo_label_dic=hypo_label_dic)
#if METHOD == "transf_embed":
df_train_format = df_train
df_test_format = df_test



##### train classifier

## ! do hp-search in separate script. should not take too long





label_text_alphabetical = np.sort(df_cl.label_text.unique())

model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=label_text_alphabetical, model_max_length=MODEL_MAX_LENGTH)


#### tokenize

dataset = tokenize_datasets(df_train_samp=df_train_format, df_test=df_test_format, tokenizer=tokenizer, method=METHOD, max_length=MODEL_MAX_LENGTH)


### create trainer

## automatically calculate roughly adequate epochs for number of data points
if METHOD == "standard_dl":
    max_steps = 7_000  # value chosen to lead to roughly 45 epochs with 5k n_data, 23 with 10k, then decrease epochs
    batch_size = 32
    #min_epochs = 10
    max_epochs = 50 #50  # good value from NLI paper experience for aroung 500 - 5k data
    n_data = len(df_train_format)
    steps_one_epoch = n_data / batch_size
    n_epochs = 0
    n_steps = 0
    while (n_epochs < max_epochs) and (n_steps < max_steps):
        n_epochs += 1
        n_steps += steps_one_epoch  # = steps_one_epoch
    print("Epochs: ", n_epochs)
    print("Steps: ", n_steps)
    HYPER_PARAMS = {'lr_scheduler_type': 'constant', 'learning_rate': 2e-5, 'num_train_epochs': n_epochs, 'seed': SEED_GLOBAL, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.06, 'weight_decay': 0.01, 'per_device_eval_batch_size': 200}  # "do_eval": False
elif METHOD == "nli":
    HYPER_PARAMS = {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 20, 'seed': SEED_GLOBAL, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.40, 'weight_decay': 0.01, 'per_device_eval_batch_size': 200}  # "do_eval": False
else:
    raise Exception("Method not implemented for hps")

# based on paper https://arxiv.org/pdf/2111.09543.pdf
if MODEL_SIZE == "large":
    HYPER_PARAMS.update({"per_device_eval_batch_size": 80, 'learning_rate': 9e-6})


## create trainer
# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa".lower() in MODEL_NAME.lower(): fp16_bool = False  # mDeBERTa does not support FP16 yet

train_args = set_train_args(hyperparams_dic=HYPER_PARAMS, training_directory=TRAINING_DIRECTORY, disable_tqdm=False, evaluation_strategy="no", fp16=fp16_bool)

trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=dataset, train_args=train_args,
                         method=METHOD, label_text_alphabetical=label_text_alphabetical)

# train
trainer.train()



### Evaluate
# test on test set
results_test = trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
print(results_test)

# do prediction on entire corpus? No.
# could do because annotations also contribute to estimate of distribution
#dataset["all"] = datasets.concatenate_datasets([dataset["test"], dataset["train"]])
#results_corpus = trainer.evaluate(eval_dataset=datasets.concatenate_datasets([dataset["train"], dataset["test"]]))  # eval_dataset=encoded_dataset["test"]
# with NLI, cannot run inference also on train set, because augmented train set can have different length than original train-set


#### prepare data for redoing figure from paper

assert (df_test["label"] == results_test["eval_label_gold_raw"]).all
df_test["label_pred"] = results_test["eval_label_predicted_raw"]

# ! careful about this - adding true labels to label_pred col for those where annotation is available - not really a prediction, but we know the pred in this context thanks to annotation
# mostly adding this to avoid downstream errors (but also reflects available knowledge for the approach)
df_train["label_pred"] = [np.nan] * len(df_train["label"])

df_cl_concat = pd.concat([df_train, df_test])

# add label text for predictions
label_text_map = {}
for i, row in df_cl_concat[~df_cl_concat.label_text.duplicated(keep='first')].iterrows():
    label_text_map.update({row["label"]: row["label_text"]})
df_cl_concat["label_text_pred"] = df_cl_concat["label_pred"].map(label_text_map)


## save data
langs_concat = "_".join(LANGUAGE_LST)

#df_cl_concat.to_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.csv", index=False)
df_cl_concat.to_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.zip",
                    compression={"method": "zip", "archive_name": f"df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.csv"}, index=False)



print("Script done.")





