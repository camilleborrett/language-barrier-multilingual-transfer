

# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
parser.add_argument('-b', '--batch_size', type=int,
                    help='batch_size for translations')
parser.add_argument('-ds', '--dataset', type=str,
                    help='string of csv dataset title, without the .csv')

parser.add_argument('-lang', '--language_target', type=str, #nargs='+',
                    help='Language to translate to')
parser.add_argument('-txt', '--text_col', type=str,
                    help='name of text column')

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
  args = parser.parse_args(["--nmt_model", "m2m_100_418M", "--language_target", "en", "--text_col", "text",  # opus-mt, "m2m_100_1.2B", "m2m_100_418M"
                            "--dataset", "df_pimpo_samp", "--batch_size", "8"])


NMT_MODEL = args.nmt_model
BATCH_SIZE = args.batch_size
DATASET = args.dataset
LANGUAGE_TARGET = args.language_target
TEXT_COL = args.text_col


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



## load df to translate
df = pd.read_csv(f"./data-clean/{DATASET}.zip", sep=",")   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',



## drop duplicates
print(len(df))
#df = df[~df.text_original.duplicated(keep='first')]
#df = df.reset_index(drop=True)

## translate each language in all other languages
# all parameters/methods for .translate here: https://github.com/UKPLab/EasyNMT/blob/main/easynmt/EasyNMT.py

# has to be M2M due to many language directions
model = EasyNMT(NMT_MODEL)  # m2m_100_418M,  m2m_100_1.2B, facebook/m2m100-12B-last-ckpt  opus-mt,

## check if translation direction available
# handles different source and target lang scenarios
if isinstance(LANGUAGE_TARGET, list):
    target_lang_lst = LANGUAGE_TARGET
else:
    target_lang_lst = [LANGUAGE_TARGET]
#target_lang_lst = ["en", "de", "es", "fr", "ko", "tr", "ru"]
source_lang_lst = df.language_iso.unique()
source_lang_available_dic = {f"source_langs_for_target_{lang_target}": model.get_languages(target_lang=lang_target) for lang_target in target_lang_lst}

trans_directions_unavailable = []
for source_lang in source_lang_lst:
    for target_lang in target_lang_lst:
        if source_lang in source_lang_available_dic[f"source_langs_for_target_{target_lang}"]:
            #print(f"Translation possible from {source_lang} to {target_lang}")
            continue
        elif source_lang == target_lang:
            continue
        else:
            #print(f"! Unavailable translation direction: from {source_lang} to {target_lang}")
            trans_directions_unavailable.append({"lang_source": source_lang, "lang_target": target_lang})

assert len(trans_directions_unavailable) < 1, f"Translations unavailable for directions: {trans_directions_unavailable}"
print(f"All translation directions are available from source: {source_lang_lst}   to target: {target_lang_lst}")


# translation for datasets with single text column
def translate_many2target(df=None, target_lang=None, batch_size=8, text_col=None):
    df_step_lst = []
    for source_lang in tqdm.tqdm(np.sort(df.language_iso.unique()).tolist(), desc="Per source language loop", leave=True, position=2):
      print(f"Translating {source_lang} to {target_lang}")
      df_step2 = df[df.language_iso == source_lang].copy(deep=True)
      print(f"    Translating {source_lang} to {target_lang}. {len(df_step2)} texts for this subset.")
      df_step2[f"{text_col}_trans"] = model.translate(df_step2[text_col].fillna('').tolist(), source_lang=source_lang, target_lang=target_lang, show_progress_bar=True, beam_size=5, batch_size=batch_size, perform_sentence_splitting=False)
      df_step_lst.append(df_step2)
      clean_memory()
    df_trans = pd.concat(df_step_lst)
    return df_trans

# translation for datasets with context sentences
def translate_many2target_context(df=None, target_lang=None, batch_size=8, text_col=None):
    df_step_lst = []
    for source_lang in tqdm.tqdm(np.sort(df.language_iso.unique()).tolist(), desc="Per source language loop", leave=True, position=2):
      print(f"Translating {source_lang} to {target_lang}")
      df_step2 = df[df.language_iso == source_lang].copy(deep=True)
      print(f"    Translating {source_lang} to {target_lang}. {len(df_step2)} texts for this subset.")
      df_step2["text_original_trans"] = model.translate(df_step2["text_original"].fillna('').tolist(), source_lang=source_lang, target_lang=target_lang, show_progress_bar=True, beam_size=5, batch_size=batch_size, perform_sentence_splitting=False)
      df_step2["text_preceding_trans"] = model.translate(df_step2["text_preceding"].fillna('').tolist(), source_lang=source_lang, target_lang=target_lang, show_progress_bar=True, beam_size=5, batch_size=batch_size, perform_sentence_splitting=False)
      df_step2["text_following_trans"] = model.translate(df_step2["text_following"].fillna('').tolist(), source_lang=source_lang, target_lang=target_lang, show_progress_bar=True, beam_size=5, batch_size=batch_size, perform_sentence_splitting=False)
      #df_step2["language_iso_trans"] = [target_lang] * len(df_step2)
      df_step_lst.append(df_step2)
      clean_memory()
    #df_step["text_original_trans"] = model.translate(df_step["text_original"].tolist(), target_lang=lang_target, show_progress_bar=True, beam_size=5, batch_size=32, perform_sentence_splitting=False)
    #df_step_lst.append(df_step)
    df_trans = pd.concat(df_step_lst)
    return df_trans


if DATASET == "df_pimpo_samp":
    df = translate_many2target_context(df=df, target_lang=LANGUAGE_TARGET, batch_size=BATCH_SIZE, text_col=None)
#elif DATASET == "dfwithonecol":
else:
    df = translate_many2target(df=df, target_lang=LANGUAGE_TARGET, batch_size=BATCH_SIZE, text_col=TEXT_COL)
#else:
#    raise Exception(f"Dataset {DATASET} not implemented")


df["language_iso_trans"] = [LANGUAGE_TARGET] * len(df)
#df = df.drop_duplicates()

# write to disk
compression_options = dict(method='zip', archive_name=f'{DATASET}_trans_{LANGUAGE_TARGET}_{NMT_MODEL}.csv')
df.to_csv(f'./data-clean/{DATASET}_trans_{LANGUAGE_TARGET}_{NMT_MODEL}.zip', compression=compression_options, index=False)
#df.to_csv(f"./data-clean/{DATASET}_trans_{LANGUAGE_TARGET}_{NMT_MODEL}.csv", index=False)


print("Script done.")


