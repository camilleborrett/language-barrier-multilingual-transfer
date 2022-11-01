

# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
parser.add_argument('-ds', '--dataset', type=str,
                    help='Which dataset?')

args = parser.parse_args()

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

df_train = pd.read_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}.csv", sep=",").reset_index(drop=True)   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',
df_test = pd.read_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}.csv", sep=",").reset_index(drop=True)   #on_bad_lines='skip' encoding='utf-8',  # low_memory=False  #lineterminator='\t',#


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
embeddings_multi_test = model_multiling.encode(df_test.text_original_trans, convert_to_tensor=False)
embeddings_multi_train = model_multiling.encode(df_train.text_original_trans, convert_to_tensor=False)

# add to df
df_test["text_original_trans_embed_multi"] = embeddings_multi_test.tolist()
df_train["text_original_trans_embed_multi"] = embeddings_multi_train.tolist()

### English embeddings - embed only english (translated) texts with English model
# test
text_en_test = df_test[df_test.language_iso_trans == "en"].text_original_trans
text_en_test_index = df_test[df_test.language_iso_trans == "en"].text_original_trans.index  # saving index for merge below
# train
text_en_train = df_train[df_train.language_iso_trans == "en"].text_original_trans
text_en_train_index = df_train[df_train.language_iso_trans == "en"].text_original_trans.index  # saving index for merge below

# embed
embeddings_en_test = model_en.encode(text_en_test.reset_index(drop=True), convert_to_tensor=False)  # have to reset index, encode method throws error otherwise
embeddings_en_train = model_en.encode(text_en_train.reset_index(drop=True), convert_to_tensor=False)  # have to reset index, encode method throws error otherwise

# merge subset of English embeddings with full df via index
embeddings_en_test_series = pd.Series(index=text_en_test_index, data=embeddings_en_test.tolist()).rename("text_original_trans_embed_en")
df_test_embed = df_test.merge(embeddings_en_test_series, how='left', left_index=True, right_index=True)

embeddings_en_train_series = pd.Series(index=text_en_train_index, data=embeddings_en_train.tolist()).rename("text_original_trans_embed_en")
df_train_embed = df_train.merge(embeddings_en_train_series, how='left', left_index=True, right_index=True)

## tests df_test
# check of lengths make sense
print(df_test.language_iso_trans.value_counts())
print(len(df_test_embed))
print(len(df_test))

# English
print(len(df_test_embed["text_original_trans_embed_en"]))

print(sum(pd.isna(df_test_embed["text_original_trans_embed_en"])))
print(len(df_test_embed[df_test_embed.language_iso_trans == "en"]["text_original_trans"]))

print(sum(pd.isna(df_test_embed["text_original_trans_embed_en"])) + len(df_test_embed[df_test_embed.language_iso_trans == "en"]["text_original_trans"]))
print(len(df_test_embed["text_original_trans"]))

# Multi
print(len(df_test_embed["text_original_trans_embed_multi"]))
print(sum(pd.isna(df_test_embed["text_original_trans_embed_multi"])))

## tests df_train
# check of lengths make sense
print(df_train.language_iso_trans.value_counts())
print(len(df_train_embed))
print(len(df_train))

# English
print(len(df_train_embed["text_original_trans_embed_en"]))

print(sum(pd.isna(df_train_embed["text_original_trans_embed_en"])))
print(len(df_train_embed[df_train_embed.language_iso_trans == "en"]["text_original_trans"]))

print(sum(pd.isna(df_train_embed["text_original_trans_embed_en"])) + len(df_train_embed[df_train_embed.language_iso_trans == "en"]["text_original_trans"]))
print(len(df_train_embed["text_original_trans"]))

# Multi
print(len(df_train_embed["text_original_trans_embed_multi"]))
print(sum(pd.isna(df_train_embed["text_original_trans_embed_multi"])))

print(df_test_embed.drop(columns=["parfam", "date", "party", "partyname"]).sample(n=100, random_state=42))

#### write to disk
df_train_embed.to_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}_embed.csv", index=False)
df_test_embed.to_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}_embed.csv", index=False)