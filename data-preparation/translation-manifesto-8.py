

# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
parser.add_argument('-b', '--batch_size', type=int,
                    help='batch_size for translations')
parser.add_argument('-ds', '--dataset', type=str,
                    help='Which dataset?')
parser.add_argument('-m_length', '--max_length', type=int,
                    help='max token length for translation')

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
  args = parser.parse_args(["--nmt_model", "m2m_100_418M",  #"m2m_100_1.2B", "m2m_100_418M"
                            "--dataset", "manifesto-8", "--batch_size", 16])


NMT_MODEL = args.nmt_model
BATCH_SIZE = args.batch_size
DATASET = args.dataset
MAX_LENGTH = args.max_length


## load packages
import pandas as pd
import numpy as np
from easynmt import EasyNMT
import tqdm  # https://github.com/tqdm/tqdm#documentation
import torch
import gc

def clean_memory():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  gc.collect()


#BATCH_SIZE = 64


## load df to translate
df_train = pd.read_csv(f"./data-clean/df_{DATASET}_train.zip", sep=",")   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',
df_test = pd.read_csv(f"./data-clean/df_{DATASET}_test.zip", sep=",")   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',

## for testing
df_train = df_train.groupby(by="language_iso").apply(lambda x: x.sample(n=min(len(x), 10), random_state=42))
df_test = df_test.groupby(by="language_iso").apply(lambda x: x.sample(n=min(len(x), 10), random_state=42))



df_train = df_train.reset_index(drop=True)  # unnecessary nested index
df_test = df_test.reset_index(drop=True)  # unnecessary nested index



## translate each language in all other languages
# all parameters/methods for .translate here: https://github.com/UKPLab/EasyNMT/blob/main/easynmt/EasyNMT.py
#lang_lst_pimpo = ["sv", "no", "da", "fi", "nl", "es", "de", "en", "fr"]
lang_lst = ["en", "de", "es", "fr", "ko", "tr", "ru"]  #["eng", "deu", "spa", "fra", "kor", "jpn", "tur", "rus"]  #"ja"

# has to be M2M due to many language directions
model = EasyNMT(NMT_MODEL)  # m2m_100_418M,  m2m_100_1.2B, facebook/m2m100-12B-last-ckpt  opus-mt,

def translate_all2all(df=None, lang_lst=None, batch_size=8):
  df_step_lst = []
  for lang_target in tqdm.tqdm(lang_lst, desc="Overall all2all translation loop", leave=True):
    df_step = df[df.language_iso != lang_target].copy(deep=True)
    print("Translating texts from all other languages to: ", lang_target, ". ", len(df_step), " texts overall.")
    # specify source language to avoid errors. Automatic language detection can (falsely) identify languages that are not supported by model.
    for lang_source in tqdm.tqdm(np.sort(df_step.language_iso.unique()).tolist(), desc="Per source language loop", leave=True, position=2):
      df_step2 = df_step[df_step.language_iso == lang_source].copy(deep=True)
      print(f"    Translating {lang_source} to {lang_target}. {len(df_step2)} texts for this subset.")
      df_step2["text_original_trans"] = model.translate(df_step2["text_original"].tolist(), source_lang=lang_source, target_lang=lang_target, show_progress_bar=False, beam_size=5, batch_size=batch_size, perform_sentence_splitting=False, max_length=MAX_LENGTH)
      df_step2["language_iso_trans"] = [lang_target] * len(df_step2)
      df_step_lst.append(df_step2)
      clean_memory()
    #df_step["text_original_trans"] = model.translate(df_step["text_original"].tolist(), target_lang=lang_target, show_progress_bar=True, beam_size=5, batch_size=32, perform_sentence_splitting=False)
    #df_step_lst.append(df_step)
  return pd.concat(df_step_lst)


## translate test
df_test_samp_trans = translate_all2all(df=df_test, lang_lst=lang_lst, batch_size=BATCH_SIZE)  # df[df.language.isin(["de", "en"])].sample(n=20, random_state=42)
# concatenate translated texts with original texts
print(len(df_test_samp_trans))
df_test["text_original_trans"] = df_test["text_original"]  #[np.nan] * len(df_test)
df_test["language_iso_trans"] = df_test["language_iso"]  #[np.nan] * len(df_test)
df_test_trans_concat = pd.concat([df_test, df_test_samp_trans], axis=0)
df_test_trans_concat = df_test_trans_concat.drop_duplicates()
print(len(df_test_trans_concat))
# write to disk
#df_test_trans_concat.to_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}.csv", index=False)
df_test_trans_concat.to_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}.zip",
                            compression={"method": "zip", "archive_name": f"df_{DATASET}_test_trans_{NMT_MODEL}.csv"},
                            index=False)


## translate test
df_train_samp_trans = translate_all2all(df=df_train, lang_lst=lang_lst, batch_size=BATCH_SIZE)  # df[df.language.isin(["de", "en"])].sample(n=20, random_state=42)
# concatenate translated texts with original texts
print(len(df_train_samp_trans))
df_train["text_original_trans"] = df_train["text_original"]  #[np.nan] * len(df_train)
df_train["language_iso_trans"] = df_train["language_iso"]  #[np.nan] * len(df_train)
df_train_trans_concat = pd.concat([df_train, df_train_samp_trans], axis=0)
df_train_trans_concat = df_train_trans_concat.drop_duplicates()
print(len(df_train_trans_concat))
# write to disk
#df_train_trans_concat.to_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}.csv", index=False)
df_train_trans_concat.to_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}.zip",
                            compression={"method": "zip", "archive_name": f"df_{DATASET}_train_trans_{NMT_MODEL}.csv"},
                            index=False)

print("Script done.")


