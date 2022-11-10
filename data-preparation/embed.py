

# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
parser.add_argument('-ds', '--dataset', type=str,
                    help='Which dataset?')


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
                            "--dataset", "manifesto-8"])


NMT_MODEL = args.nmt_model
DATASET = args.dataset


#set wd
import os
print(os.getcwd())
#os.chdir("/content/drive/My Drive/PhD/multilingual-paper")
#print(os.getcwd())


"""### code to embed text (en & multiling)"""

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
import torch

df = pd.read_csv(f"./data-clean/df_{DATASET}_trans_{NMT_MODEL}.zip", sep=",").reset_index(drop=True)   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',

# if translation failed and produced nan, add empty string - to avoid errors - don't delete to maintain length of df
df["text_original_trans"] = df.text_original_trans.fillna("[TRANS_FAIL]")
# reset index because sentence transformers needs sequential index
df = df.reset_index(drop=True)



## load model
# https://sbert.net/docs/pretrained_models.html
# used by Hauke: "sentence-transformers/paraphrase-xlm-r-multilingual-v1" ; probably best: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" ; best for fast testing  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# multiling model
model_multiling = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")  # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_multiling.to(device)
# en model
model_en = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")  #'all-mpnet-base-v2'  # "sentence-transformers/paraphrase-MiniLM-L12-v2", "sentence-transformers/paraphrase-mpnet-base-v2"
model_en.to(device)

### multilingual embeddings - simply on text_original
# embedding the translated column here to test if the slight variations from translated texts are useful for augmentation
embeddings_multi = model_multiling.encode(df.text_original_trans, convert_to_tensor=False)
# add to df
df["text_original_trans_embed_multi"] = embeddings_multi.tolist()

### English embeddings - embed only english (translated) texts with English model
text_en = df[df.language_iso_trans == "en"].text_original_trans
text_en_index = df[df.language_iso_trans == "en"].text_original_trans.index  # saving index for merge below
# embed
embeddings_en = model_en.encode(text_en.reset_index(drop=True), convert_to_tensor=False)  # have to reset index, encode method throws error otherwise

# merge subset of English embeddings with full df via index
embeddings_en_series = pd.Series(index=text_en_index, data=embeddings_en.tolist()).rename("text_original_trans_embed_en")
df_embed = df.merge(embeddings_en_series, how='left', left_index=True, right_index=True)


## tests df
# check of lengths make sense
print(df.language_iso_trans.value_counts())
print(len(df_embed))
print(len(df))

# English
print(len(df_embed["text_original_trans_embed_en"]))

print(sum(pd.isna(df_embed["text_original_trans_embed_en"])))
print(len(df_embed[df_embed.language_iso_trans == "en"]["text_original_trans"]))

print(sum(pd.isna(df_embed["text_original_trans_embed_en"])) + len(df_embed[df_embed.language_iso_trans == "en"]["text_original_trans"]))
print(len(df_embed["text_original_trans"]))

# Multi
print(len(df_embed["text_original_trans_embed_multi"]))
print(sum(pd.isna(df_embed["text_original_trans_embed_multi"])))

print(df_embed.drop(columns=["parfam", "date", "party", "partyname"]).sample(n=100, random_state=42))


#### write to disk
df_embed.to_csv(f"./data-clean/df_{DATASET}_trans_{NMT_MODEL}_embed.zip",
                compression={"method": "zip", "archive_name": f"df_{DATASET}_trans_{NMT_MODEL}_embed.csv"}, index=False)


print("Script done.")
