

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
df = pd.read_csv("./data-clean/df_manifesto_test.csv", sep=",")   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',
df = df.reset_index(drop=True)  # unnecessary nested index

## take sample for testing and to reduce excessive compute
df_samp = df.groupby(by="language_iso").apply(lambda x: x.groupby(by="label_subcat_text").apply(lambda x: x.sample(n=min(len(x), 50), random_state=42)))
print(len(df_samp))
print(df_samp.language_iso.value_counts())


## translate each language in all other languages
# all parameters/methods for .translate here: https://github.com/UKPLab/EasyNMT/blob/main/easynmt/EasyNMT.py
#lang_lst_pimpo = ["sv", "no", "da", "fi", "nl", "es", "de", "en", "fr"]
lang_lst = ["en", "de", "es", "fr", "ko", "ja", "tr", "ru"]  #["eng", "deu", "spa", "fra", "kor", "jpn", "tur", "rus"]

model = EasyNMT('m2m_100_418M')  # m2m_100_418M, opus-mt, m2m100_1.2B, facebook/m2m100-12B-last-ckpt

def translate_all2all(df=None, lang_lst=None):
  df_step_lst = []
  for lang_target in tqdm.tqdm(lang_lst, desc="Overall all2all translation loop", leave=True):
    df_step = df[df.language_iso != lang_target].copy(deep=True)
    print("Translating texts from all other languages to: ", lang_target, ". ", len(df_step), " texts overall.")
    # specify source language to avoid errors. Automatic language detection can (falsely) identify languages that are not supported by model.
    for lang_source in tqdm.tqdm(np.sort(df_step.language_iso.unique()).tolist(), desc="Per source language loop", leave=True, position=2):
      df_step2 = df_step[df_step.language_iso == lang_source].copy(deep=True)
      print(f"    Translating {lang_source} to {lang_target}. {len(df_step2)} texts for this subset.")
      df_step2["text_original_trans"] = model.translate(df_step2["text_original"].tolist(), source_lang=lang_source, target_lang=lang_target, show_progress_bar=False, beam_size=5, batch_size=56, perform_sentence_splitting=False)
      df_step2["language_iso_trans"] = [lang_target] * len(df_step2)
      df_step_lst.append(df_step2)
      clean_memory()
    #df_step["text_original_trans"] = model.translate(df_step["text_original"].tolist(), target_lang=lang_target, show_progress_bar=True, beam_size=5, batch_size=32, perform_sentence_splitting=False)
    #df_step_lst.append(df_step)
  return pd.concat(df_step_lst)


df_trans = translate_all2all(df=df_samp, lang_lst=lang_lst)  # df[df.language.isin(["de", "en"])].sample(n=20, random_state=42)

## concatenate translated texts with original texts
# careful about column mismatches
print(len(df_trans))
df_samp["text_original_trans"] = df_samp["text_original"]  #[np.nan] * len(df_samp)
df_samp["language_iso_trans"] = df_samp["language_iso"]  #[np.nan] * len(df_samp)
df_trans_concat = pd.concat([df_samp, df_trans], axis=0)
df_trans_concat = df_trans_concat.drop_duplicates()
print(len(df_trans_concat))


#### write to disk
df_trans_concat.to_csv("./data-clean/df_manifesto_test_trans.csv", index=False)


### tests
#df_trans_concat = pd.read_csv("./data-clean/df_manifesto_train_trans.csv", sep=",")   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',

#len(df.sentence_id.value_counts())
#len(df_samp.sentence_id.value_counts())



