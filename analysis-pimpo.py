




# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

# main args
parser.add_argument('-lang', '--languages', type=str, nargs='+',
                    help='List of languages to iterate over.')
parser.add_argument('-samp', '--sample', type=int, #nargs='+',
                    help='Sample')
parser.add_argument('-max_e', '--max_epochs', type=int, #nargs='+',
                    help='number of epochs')
parser.add_argument('-date', '--study_date', type=str,
                    help='Date')
parser.add_argument('-t', '--task', type=str,
                    help='task about integration or immigration?')

## maybe to use later
"""parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model or "SVM" or "logistic". Must fit to "method" argument.')
parser.add_argument('-augment', '--augmentation_nmt', type=str,
                    help='Whether and how to augment the data with machine translation (MT).')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings-en" or "embeddings-multi"')
parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
"""

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
  args = parser.parse_args(["--languages", "en", "de", "--max_epochs", "30",
                            "--sample", "500", "--study_date", "221103"])

LANGUAGE_LST = args.languages
SAMPLE_PER_LANG = args.sample
DATE = args.study_date
MAX_EPOCHS = args.max_epochs
TASK = args.task


## load relevant packages
import pandas as pd
import numpy as np
import os
import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments

# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)
from helpers import compute_metrics_standard, clean_memory
from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer


## set main arguments
SEED_GLOBAL = 42
DATASET = "pimpo"
#HYPER_PARAMS = {'lr_scheduler_type': 'constant', 'learning_rate': 2e-5, 'num_train_epochs': 50, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 200}
TRAINING_DIRECTORY = f"results/{DATASET}"
METHOD = "standard_dl"
MODEL_NAME = "microsoft/mdeberta-v3-base"  # "microsoft/Multilingual-MiniLM-L12-H384", microsoft/mdeberta-v3-base
MODEL_MAX_LENGTH = 256

# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa".lower() in MODEL_NAME.lower(): fp16_bool = False  # mDeBERTa does not support FP16 yet



## load dataset
df = pd.read_csv("./data-clean/df_pimpo_all.csv")


### select training data
# note that for this substantive use-case, it seems fine to predict also on training data,
# because it is about the substantive outcome of the approach, not out-of-sample accuracy

# choose bigger text window to improve performance and imitate annotation input
df["text_prep"] = df["text_preceding"].fillna('') + " " + df["text_original"] + " " + df["text_following"].fillna('')

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
sample_n = 20_000
df_cl = df_cl.groupby(by="label_text", as_index=False, group_keys=False).apply(lambda x: x.sample(n=sample_n, random_state=42) if x.label_text.iloc[0] == "no_topic" else x)
df_cl["label_text"].value_counts()

## select training data
# via language
#df_train = df_cl[df_cl.language_iso.isin(["en"])]
#df_train = df_cl[df_cl.language_iso.isin(["en", "de"])]
#df_train = df_cl[df_cl.language_iso.isin(["en", "de", "sv", "fr"])]
df_train = df_cl[df_cl.language_iso.isin(LANGUAGE_LST)]

# take sample for all topical labels - should not be more than SAMPLE (e.g. 500) per language to simulate realworld situation and prevent that adding some languages adds much more data than adding other languages - theoretically each coder can code n SAMPLE data
task_label_text_wo_notopic = task_label_text[:3]
df_train_samp1 = df_train.groupby(by="language_iso", as_index=False, group_keys=False).apply(lambda x: x[x.label_text.isin(task_label_text_wo_notopic)].sample(n=min(len(x[x.label_text.isin(task_label_text_wo_notopic)]), SAMPLE_PER_LANG), random_state=42))
df_train_samp2 = df_train.groupby(by="language_iso", as_index=False, group_keys=False).apply(lambda x: x[x.label_text == "no_topic"].sample(n=min(len(x[x.label_text == "no_topic"]), SAMPLE_PER_LANG), random_state=42))
df_train = pd.concat([df_train_samp1, df_train_samp2])

# reduce n no_topic data
df_train_topic = df_train[df_train.label_text != "no_topic"]
df_train_no_topic = df_train[df_train.label_text == "no_topic"].sample(n=len(df_train_topic), random_state=SEED_GLOBAL)
df_train = pd.concat([df_train_topic, df_train_no_topic])

# avoid random overlap between topic and no-topic ?
# ! not sure if this adds value - maybe I even want overlap to teach it to look only at the middle sentence? Or need to check overlap by 3 text columns ?
#print("Is there overlapping text between train and test?", df_train_no_topic[df_train_no_topic.text_original.isin(df_train_topic.text_prep)])

## create df test
# just to get accuracy figure as benchmark, less relevant for substantive use-case
df_test = df_cl[~df_cl.index.isin(df_train.index)]
assert len(df_train) + len(df_test) == len(df_cl)


##### train classifier

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, model_max_length=MODEL_MAX_LENGTH);
# define config. label text to label id in alphabetical order
label_text_alphabetical = np.sort(df_cl.label_text.unique())
label2id = dict(zip(np.sort(label_text_alphabetical), np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist()))  # .astype(int).tolist()
id2label = dict(zip(np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist(), np.sort(label_text_alphabetical)))
config = AutoConfig.from_pretrained(MODEL_NAME, label2id=label2id, id2label=id2label);
# load model with config
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config);
# to device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device);


## tokenize
import datasets

# train, val, test all in one datasetdict:
# inference on entire corpus, intentionally including df_train
dataset = datasets.DatasetDict({"train": datasets.Dataset.from_pandas(df_train),
                                "test": datasets.Dataset.from_pandas(df_test)})

#def tokenize_func_nli(examples):
#    return tokenizer(examples["text_prepared"], examples["hypothesis"], truncation=True, max_length=max_length)  # max_length=512,  padding=True
def tokenize_func_mono(examples):
    return tokenizer(examples["text_prep"], truncation=True, max_length=MODEL_MAX_LENGTH)  # max_length=512,  padding=True

dataset["train"] = dataset["train"].map(tokenize_func_mono, batched=True)  # batch_size=len(df_train)
dataset["test"] = dataset["test"].map(tokenize_func_mono, batched=True)  # batch_size=len(df_train)

### create trainer

## automatically calculate roughly adequate epochs for number of data points
max_steps = 7_000  # value chosen to lead to roughly 45 epochs with 5k n_data, 23 with 10k, then decrease epochs
batch_size = 32
#min_epochs = 10
max_epochs = MAX_EPOCHS #50  # good value from NLI paper experience for aroung 500 - 5k data
n_data = len(df_train)
steps_one_epoch = n_data / batch_size

n_epochs = 0
n_steps = 0
while (n_epochs < max_epochs) and (n_steps < max_steps):
    n_epochs += 1
    n_steps += steps_one_epoch  # = steps_one_epoch
print("Epochs: ", n_epochs)
print("Steps: ", n_steps)

HYPER_PARAMS = {'lr_scheduler_type': 'constant', 'learning_rate': 2e-5, 'num_train_epochs': n_epochs, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 200}


train_args = set_train_args(hyperparams_dic=HYPER_PARAMS, training_directory=TRAINING_DIRECTORY, disable_tqdm=False, evaluation_strategy="no", fp16=fp16_bool)

trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=dataset, train_args=train_args,
                         method=METHOD, label_text_alphabetical=label_text_alphabetical)

# train
trainer.train()



### Evaluate
# test on test set
results_test = trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
print(results_test)

# do prediction on entire corpus
#dataset["all"] = datasets.concatenate_datasets([dataset["test"], dataset["train"]])
results_corpus = trainer.evaluate(eval_dataset=datasets.concatenate_datasets([dataset["train"], dataset["test"]]))  # eval_dataset=encoded_dataset["test"]



#### prepare data for redoing figure from paper
df_cl_concat = pd.concat([df_train, df_test])

## add predicted labels to df
assert (df_cl_concat["label"] == results_corpus["eval_label_gold_raw"]).all
df_cl_concat["label_pred"] = results_corpus["eval_label_predicted_raw"]

# add label text for predictions
label_text_map = {}
for i, row in df_cl_concat[~df_cl_concat.label_text.duplicated(keep='first')].iterrows():
    label_text_map.update({row["label"]: row["label_text"]})
df_cl_concat["label_pred_text"] = df_cl_concat["label_pred"].map(label_text_map)


## save data
langs_concat = "_".join(LANGUAGE_LST)

df_cl_concat.to_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{SAMPLE_PER_LANG}samp_{langs_concat}_{DATE}.csv", index=False)


print("Script done.")





