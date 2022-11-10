




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
parser.add_argument('-meth', '--method', type=str,
                    help='NLI or standard dl?')
parser.add_argument('-v', '--vectorizer', type=str,
                    help='en or multi?')
parser.add_argument('-hypo', '--hypothesis', type=str,
                    help='which hypothesis?')

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
  args = parser.parse_args(["--languages", "en", "de", "--max_epochs", "2", "--task", "immigration", "--vectorizer", "en"
                            "--sample", "10", "--study_date", "221103"])

LANGUAGE_LST = args.languages
SAMPLE_PER_LANG = args.sample
DATE = args.study_date
MAX_EPOCHS = args.max_epochs
TASK = args.task
METHOD = args.method
VECTORIZER = args.vectorizer
HYPOTHESIS = args.hypothesis

SAMPLE_NO_TOPIC = 10_000
SAMPLE_DF_TEST = 10_000

## set main arguments
SEED_GLOBAL = 42
DATASET = "pimpo"
#HYPER_PARAMS = {'lr_scheduler_type': 'constant', 'learning_rate': 2e-5, 'num_train_epochs': 50, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 200}
TRAINING_DIRECTORY = f"results/{DATASET}"

if (VECTORIZER == "multi") and (METHOD == "nli"):
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"  #"microsoft/mdeberta-v3-base"  "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"  # "microsoft/Multilingual-MiniLM-L12-H384", microsoft/mdeberta-v3-base
elif (VECTORIZER == "en") and (METHOD == "nli"):
    MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"  # microsoft/deberta-v3-large  "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
elif (VECTORIZER == "multi") and (METHOD == "standard_dl"):
    MODEL_NAME = "microsoft/mdeberta-v3-base"  # microsoft/deberta-v3-large  "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
elif (VECTORIZER == "en") and (METHOD == "standard_dl"):
    MODEL_NAME = "microsoft/deberta-v3-large"  # microsoft/deberta-v3-large  "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
else:
    raise Exception(f"VECTORIZER {VECTORIZER} or METHOD {METHOD} not implemented")

MODEL_MAX_LENGTH = 256



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
from helpers import compute_metrics_standard, clean_memory, compute_metrics_nli_binary
from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer, format_nli_trainset, format_nli_testset


# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa".lower() in MODEL_NAME.lower(): fp16_bool = False  # mDeBERTa does not support FP16 yet



##### load dataset
df = pd.read_csv("./data-clean/df_pimpo_samp_trans_m2m_100_1.2B.csv")

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
    if METHOD == "standard_dl":
        df["text_prepared"] = df["text_preceding"].fillna('') + " " + df["text_original"] + " " + df["text_following"].fillna('')
    elif METHOD == "nli":
        df["text_prepared"] = df["text_preceding"].fillna('') + '. The quote: "' + df["text_original"] + '". ' + df["text_following"].fillna('')
elif VECTORIZER == "en":
    if METHOD == "standard_dl":
        df["text_prepared"] = df["text_preceding_trans"].fillna('') + ' ' + df["text_original_trans"] + ' ' + df["text_following_trans"].fillna('')
    elif METHOD == "nli":
        df["text_prepared"] = df["text_preceding_trans"].fillna('') + '. The quote: "' + df["text_original_trans"] + '". ' + df["text_following_trans"].fillna('')
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
df_train_samp1 = df_train.groupby(by="language_iso", as_index=False, group_keys=False).apply(lambda x: x[x.label_text.isin(task_label_text_wo_notopic)].sample(n=min(len(x[x.label_text.isin(task_label_text_wo_notopic)]), SAMPLE_PER_LANG), random_state=42))
df_train_samp2 = df_train.groupby(by="language_iso", as_index=False, group_keys=False).apply(lambda x: x[x.label_text == "no_topic"].sample(n=min(len(x[x.label_text == "no_topic"]), SAMPLE_PER_LANG), random_state=42))
df_train = pd.concat([df_train_samp1, df_train_samp2])

# reduce n no_topic data
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
df_test = df_test.sample(n=min(SAMPLE_DF_TEST, len(df_test)), random_state=SEED_GLOBAL)


### format data if NLI

if TASK == "immigration":
    if HYPOTHESIS == "short":
        hypo_label_dic = {
            "immigration_neutral": "The quote is neutral towards immigration or describes the status quo of immigration.",
            "immigration_sceptical": "The quote is sceptical of immigration.",
            "immigration_supportive": "The quote is supportive of immigration.",
            "no_topic": "The quote is not about immigration.",
        }
    elif HYPOTHESIS == "long":
        hypo_label_dic = {
            "immigration_neutral": "The quote describes immigration neutrally without implied value judgement or describes the status quo of immigration, for example only stating facts or using technocratic language about immigration",
            "immigration_sceptical": "The quote describes immigration sceptically / disapprovingly. For example, the quote could mention the costs of immigration, be against migrant workers, state that foreign labour decreases natives' wages, that there are already enough refugees, refugees are actually economic migrants, be in favour of stricter immigration controls, exceptions to the freedom of movement in the EU.",
            "immigration_supportive": "The quote describes immigration favourably / supportively. For example, the quote could mention the benefits of immigration, the need for migrant workers, international obligations to take in refugees, protection of human rights, in favour of family reunification or freedom of movement in the EU.",
            "no_topic": "The quote is not about immigration.",
        }
    else:
        raise Exception(f"Hypothesis {HYPOTHESIS} not implemented")
elif TASK == "integration":
    if HYPOTHESIS == "short":
        hypo_label_dic = {
            "integration_neutral": "The quote is neutral towards immigrant integration or describes the status quo of immigrant integration.",
            "integration_sceptical": "The quote is sceptical of immigrant integration.",
            "integration_supportive": "The quote is supportive of immigrant integration.",
            "no_topic": "The quote is not about immigrant integration.",
        }
    elif HYPOTHESIS == "long":
        raise Exception("Not implemented hypo long for integration")
        """hypo_label_dic = {
            "immigration_neutral": "The quote describes immigrant integration neutrally or describes the status quo of immigrant integration, for example only stating facts or using technocratic language about immigrant integration",
            "immigration_sceptical": "The quote describes immigrant integration sceptically / disapprovingly.",
            "immigration_supportive": "The quote describes immigrant integration favourably / supportively.",
            "no_topic": "The quote is not about immigrant integration.",
        }"""
    else:
        raise Exception(f"Hypothesis {HYPOTHESIS} not implemented")
else:
    raise Exception(f"Task {TASK} not implemented")


if METHOD == "standard_dl":
    df_train_format = df_train
    df_test_format = df_test
elif METHOD == "nli":
    df_train_format = format_nli_trainset(df_train=df_train, hypo_label_dic=hypo_label_dic, random_seed=42)
    df_test_format = format_nli_testset(df_test=df_test, hypo_label_dic=hypo_label_dic)


##### train classifier
label_text_alphabetical = np.sort(df_cl.label_text.unique())

model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=label_text_alphabetical, model_max_length=MODEL_MAX_LENGTH)

"""tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, model_max_length=MODEL_MAX_LENGTH);
# define config. label text to label id in alphabetical order
label2id = dict(zip(np.sort(label_text_alphabetical), np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist()))  # .astype(int).tolist()
id2label = dict(zip(np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist(), np.sort(label_text_alphabetical)))
config = AutoConfig.from_pretrained(MODEL_NAME, label2id=label2id, id2label=id2label);
# load model with config
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config);
# to device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device);"""


#### tokenize
import datasets

dataset = tokenize_datasets(df_train_samp=df_train_format, df_test=df_test_format, tokenizer=tokenizer, method=METHOD, max_length=MODEL_MAX_LENGTH)

"""# train, val, test all in one datasetdict:
# inference on entire corpus, intentionally including df_train_format
dataset = datasets.DatasetDict({"train": datasets.Dataset.from_pandas(df_train_format),
                                "test": datasets.Dataset.from_pandas(df_test_format)})
#def tokenize_func_nli(examples):
#    return tokenizer(examples["text_prepared"], examples["hypothesis"], truncation=True, max_length=max_length)  # max_length=512,  padding=True
def tokenize_func_mono(examples):
    return tokenizer(examples["text_prepared"], truncation=True, max_length=MODEL_MAX_LENGTH)  # max_length=512,  padding=True

dataset["train"] = dataset["train"].map(tokenize_func_mono, batched=True)  # batch_size=len(df_train_format)
dataset["test"] = dataset["test"].map(tokenize_func_mono, batched=True)  # batch_size=len(df_train_format)"""



### create trainer

## automatically calculate roughly adequate epochs for number of data points
if METHOD == "standard_dl":
    max_steps = 7_000  # value chosen to lead to roughly 45 epochs with 5k n_data, 23 with 10k, then decrease epochs
    batch_size = 32
    #min_epochs = 10
    max_epochs = MAX_EPOCHS #50  # good value from NLI paper experience for aroung 500 - 5k data
    n_data = len(df_train_format)
    steps_one_epoch = n_data / batch_size
    n_epochs = 0
    n_steps = 0
    while (n_epochs < max_epochs) and (n_steps < max_steps):
        n_epochs += 1
        n_steps += steps_one_epoch  # = steps_one_epoch
    print("Epochs: ", n_epochs)
    print("Steps: ", n_steps)
    HYPER_PARAMS = {'lr_scheduler_type': 'constant', 'learning_rate': 2e-5, 'num_train_epochs': n_epochs, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.06, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128}
elif METHOD == "nli":
    HYPER_PARAMS = {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 20, 'seed': 42, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.40, 'weight_decay': 0.05, 'per_device_eval_batch_size': 128}
else:
    raise Exception("Method not implemented for hps")



## create trainer
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
#results_corpus = trainer.evaluate(eval_dataset=datasets.concatenate_datasets([dataset["train"], dataset["test"]]))  # eval_dataset=encoded_dataset["test"]

# ! with NLI, cannot run inference also on train set, because augmented train set can have different length than original train-set
# no problem, because including train set is not important, already have true predictions


#### prepare data for redoing figure from paper

## add predicted labels to df
#df_cl_concat = pd.concat([df_train, df_test])
#assert (df_cl_concat["label"] == results_corpus["eval_label_gold_raw"]).all
#df_cl_concat["label_pred"] = results_corpus["eval_label_predicted_raw"]

assert (df_test["label"] == results_test["eval_label_gold_raw"]).all
df_test["label_pred"] = results_test["eval_label_predicted_raw"]

# ! be sure to discard this downstream  ! with this I cannot use the training language's predicted label column !
# only doing this to avoid downstream errors through nan or different length
df_train["label_pred"] = df_train["label"]

df_cl_concat = pd.concat([df_train, df_test])


# add label text for predictions
label_text_map = {}
for i, row in df_cl_concat[~df_cl_concat.label_text.duplicated(keep='first')].iterrows():
    label_text_map.update({row["label"]: row["label_text"]})
df_cl_concat["label_pred_text"] = df_cl_concat["label_pred"].map(label_text_map)


## save data
langs_concat = "_".join(LANGUAGE_LST)

df_cl_concat.to_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{HYPOTHESIS}_{VECTORIZER}_{SAMPLE_PER_LANG}samp_{langs_concat}_{DATE}.csv", index=False)


print("Script done.")





