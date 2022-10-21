#!/usr/bin/env python
# coding: utf-8

### dataset not used in the end because already enough manifesto datasets

# ## Install and load packages
import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
import random
import os

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


# ## Load & prepare data

#set wd
print(os.getcwd())
#os.chdir("/content/drive/My Drive/Colab Notebooks")
print(os.getcwd())


## load data
# PImPo codebook: https://manifesto-project.wzb.eu/down/datasets/pimpo/PImPo_codebook.pdf
df = pd.read_csv("https://raw.githubusercontent.com/farzamfan/Multilingual-StD-Pipeline/main/Plm%20Manifest/PImPo%20dataset/PImPo_verbatim.csv", sep=",",  #encoding='utf-8',  # low_memory=False  #lineterminator='\t',
                 on_bad_lines='skip'
)
print(df.columns)
print(len(df))



### data cleaning
# multilingual-repo book https://manifesto-project.wzb.eu/down/datasets/pimpo/PImPo_codebook.pdf
# rn variable  "A running number sorting all quasi-sentences within each document into the order in which they appeared in the document."
# pos_corpus "Comment: Gives the position of the quasi-sentence within each document in the Manifesto Corpus. It can be used to merge the original verbatim of the quasi-sentences to the dataset. It is missing for 234 quasi-sentences from the Finnish National Coalition in 2007 and one quasi-sentence from the German Greens in 2013. For the crowd coding we have worked with the beta version of the Manifesto Corpus. The respective quasi- sentences were in the beta version, but are not in the publicly available Manifesto Corpus, because they were classified as text in margin by the Manifesto Project. The R-Script provided on the website makes it possible to add the verbatim from these quasi-sentences nonetheless."
df_cl = df.rename(columns={"content": "text"}).copy(deep=True)

## exploring multilingual data
country_map = {
    11: "swe", 12: "nor", 13: "dnk", 14: "fin", 22: "nld",
    33: "esp", 41: "deu", 42: "aut", 43: "che",
    53: "irl", 61: "usa", 62: "can", 63: "aus", 64: "nzl"
}
df_cl["country_iso"] = [country_map[country_id] for country_id in df_cl.country]
# languages ISO 639-2/T https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
languages_map = {
    11: "swe", 12: "nor", 13: "dan", 14: "fin", 22: "nld",
    33: "spa", 41: "deu", 42: "deu", 43: "deu",  # manually checked Switzerland, only has deu texts
    53: "eng", 61: "eng", 62: "fra", 63: "eng", 64: "eng"  # manually checked Canada, only has fra texts
}
df_cl["language_iso"] = [languages_map[country_id] for country_id in df_cl.country]

df_cl.language_iso.value_counts()

## select languages to analyse
#df_cl = df_cl[df_cl.language_iso.isin(["eng", "deu", "fra", "spa"])]












### adding preceding / following text column
n_unique_doc_lst = []
n_unique_doc = 0
text_preceding = []
text_following = []
for name_group, df_group in df_cl.groupby(by=["party", "date"], sort=False):  # over each doc to avoid merging sentences accross manifestos
    n_unique_doc += 1
    df_group = df_group.reset_index(drop=True)  # reset index to enable iterating over index
    for i in range(len(df_group["text"])):
        if i > 0 and i < len(df_group["text"]) - 1:
            text_preceding.append(df_group["text"][i-1])
            text_following.append(df_group["text"][i+1])
        elif i == 0:  # for very first sentence of each manifesto
            text_preceding.append("")
            text_following.append(df_group["text"][i+1])
        elif i == len(df_group["text"]) - 1:  # for last sentence
            text_preceding.append(df_group["text"][i-1])
            text_following.append("")
        else:
          raise Exception("Issue: condition not in multilingual-repo")
    n_unique_doc_lst.append([n_unique_doc] * len(df_group["text"]))
n_unique_doc_lst = [item for sublist in n_unique_doc_lst for item in sublist]

# create new columns
df_cl["text_original"] = df_cl["text"]
df_cl = df_cl.drop(columns=["text"])
df_cl["text_preceding"] = text_preceding
df_cl["text_following"] = text_following
df_cl["doc_id"] = n_unique_doc_lst  # column with unique doc identifier



## select relevant columns
df_cl = df_cl[['country_iso', 'language_iso', "doc_id", # 'gs_1r', 'gs_2r', 'date', 'party', 'pos_corpus'
               #'gs_answer_1r', 'gs_answer_2q', 'gs_answer_3q', # 'num_codings_1r', 'num_codings_2r'
              'selection', 'certainty_selection', 'topic', 'certainty_topic',
              'direction', 'certainty_direction', 'rn', 'cmp_code',  # 'manually_coded'
              'text_original', "text_preceding", "text_following"]]




### key variables:
# df_cl.selection: "Variable takes the value of 1 if immigration or immigrant integration related, and zero otherwise."
# df_cl.topic: 1 == immigration, 2 == integration, NaN == not related
# df_cl.direction: -1 == sceptical, 0 == neutral, 1 == supportive, NaN == not related to topics. "Gives the direction of a quasi-sentences as either sceptical, neutral or supportive. Missing for quasi-sentence which are classified as not-related to immigration or integration."

### gold answer variables (usec to test crowd coders)
## df_cl.gs_answer_1r == gold answers whether related to immigration or integration  used for testing crowd coders
# 0 == not immi/inti related; 1 == Immi/inti related  # "Comment: The variable gives the value of how the respective gold-sentence was coded by the authors, i.e. 0 if it was classified as not related to immigration or integration and 1 if it was regarded as related to one of these. It is missing if a quasi-sentence was not a gold sentence in the first round."
## df_cl.gs_answer_2q == Gold answers whether it is about immigration or integration or none of the two
# 1 == immigration, 2 == integration, NaN == none of the two
## df_cl.gs_answer_3q == gold answers supportive or sceptical or neutral towards topic
# -1 == sceptical, 1 == supportive, NaN == none of the two (probably neutral if also about topic)
#df_cl_gold = df_cl[['gs_answer_1r', 'gs_answer_2q', 'gs_answer_3q']]

#df_cl = df_cl[['selection', 'topic', 'direction', "text_preceding", 'text_original', "text_following"]]

## only selected texts
#df_cl[df_cl.selection != 0]
#df_cl[df_cl.manually_coded == True]

df_cl = df_cl.reset_index(drop=True)




## add gold label text column
# ! gold answer can diverge from crowd answer (e.g. index 232735). 
# ! also: if sth was gold answer for r1, it's not necessarily gold answer for r2. no gold answer for topic neutral was provided, those three where only gold for r1 (!! one of them even has divergence between gold and crowd, index 232735)

label_text_gold_lst = []
for i, row in df_cl.iterrows():
  if row["selection"] == 0:   
    label_text_gold_lst.append("no_topic")
  elif row["selection"] == 1 and row["topic"] == 1 and row["direction"] == 1:   
    label_text_gold_lst.append("immigration_supportive")
  elif row["selection"] == 1 and row["topic"] == 1 and row["direction"] == -1:   
    label_text_gold_lst.append("immigration_sceptical")
  elif row["selection"] == 1 and row["topic"] == 1 and row["direction"] == 0:   
    label_text_gold_lst.append("immigration_neutral")
  elif row["selection"] == 1 and row["topic"] == 2 and row["direction"] == 1:   
    label_text_gold_lst.append("integration_supportive")
  elif row["selection"] == 1 and row["topic"] == 2 and row["direction"] == -1:   
    label_text_gold_lst.append("integration_sceptical")
  elif row["selection"] == 1 and row["topic"] == 2 and row["direction"] == 0:       
    label_text_gold_lst.append("integration_neutral")

df_cl["label_text"] = label_text_gold_lst

df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]



df_cl[(df_cl.language != "asdf") & (df_cl.selection == 1)]




## test how many sentences have same type as preceding / following sentence
#df_test = df_cl[df_cl.label_text != "no_topic"]

test_lst = []
test_lst2 = []
test_lst_after = []
for name_df, group_df in df_cl[df_cl.language == "eng"].groupby(by="doc_id", group_keys=False, as_index=False, sort=False):
  for i in range(len(group_df)):
    # one preceding text
    if i == 0 or group_df["label_text"].iloc[i] == "no_topic":
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1]:
      test_lst.append("same_before")
    else:
      test_lst.append(f"different label before: {group_df['label_text'].iloc[i-1]}")
    # two preceding texts
    """if i < 2 or group_df["label_text"].iloc[i] == "no_topic":
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1] == group_df["label_text"].iloc[i-2]:
      test_lst2.append("same_two_before")
    else:
      test_lst2.append("different_two_before")"""
    # for following texts
    if i >= len(group_df)-1 or group_df["label_text"].iloc[i] == "no_topic":
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i+1]:
      test_lst_after.append("same_after")
    else:
      test_lst_after.append(f"different label after: {group_df['label_text'].iloc[i+1]}")

print(pd.Series(test_lst).value_counts(normalize=True), "\n")  # SOTU: 75 % of sentences have the same type as the preceeding sentence
print(pd.Series(test_lst_after).value_counts(normalize=True), "\n")  
## English
# in 50% of cases, labeled text (excluding no_topic texts) is preceded or followed by no_topic text
# in 25% by same label and in 25% by different label (other than no_topic) => no unfair advantage through data leakage if random preceding and following text added!
## Multilingual: 
# in 38% of cases, labeled text (excluding no_topic texts) is preceded or followed by no_topic text
# in 34% by same label and in 28% by different label (other than no_topic) => no unfair advantage through data leakage if random preceding and following text added!




df_cl[df_cl.selection != 0].country_name.value_counts()

print(df_cl[df_cl.selection != 0].language.value_counts())

#print(df_cl[(df_cl.language != "asdf") & (df_cl.selection != 1234)].label_text.value_counts(), "\n")

for lang in df_cl.language.unique():
  print(lang)
  print(df_cl[(df_cl.language == lang) & (df_cl.selection != 1234)].label_text.value_counts(), "\n")



## XNLI: "English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu" https://arxiv.org/pdf/1809.05053.pdf
# not in XNLI, but in PImPo: nld, nor, dan, swe, fin  -  languages in XNLI & PImPo: deu, spa, eng, fra; 




#### decide on language
#df_cl = df_cl[df_cl.language == "eng"]




### separate train and test here

## use gold set from dataset creators as training set? Issue: three categories only 1-3 examples & for integration_neutral, one of two gold does not even agree with the crowd
#df_train = df_cl[~df_cl['gs_answer_1r'].isna() | ~df_cl['gs_answer_2q'].isna() | ~df_cl['gs_answer_3q'].isna()]  #'gs_answer_2q', 'gs_answer_3q'
#df_test = df_cl[~df_cl.index.isin(df_train.index)]

## random balanced sample - fixed
# only use max 20 per class for pimpo, because overall so few examples and can still simulate easily manually created dataset. can take 100/200 for no_topic
df_train = df_cl.groupby(by=["label_text", "language"], group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=(min(int(len(x)/2), 20)), random_state=SEED_GLOBAL) if x.label_text.iloc[0] != "no_topic" 
                                                                                              else x.sample(n=(min(int(len(x)/2), 50)), random_state=SEED_GLOBAL))
# ! train only on English?
#df_train = df_train[df_train.language == "eng"]

df_test = df_cl[~df_cl.index.isin(df_train.index)]

assert len(df_train) + len(df_test) == len(df_cl)

# sample of no-topic test set for faster testing
# ! need to introduce balancing of languages here somehow ?
df_test = df_test.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=(min(len(x), 1000)), random_state=SEED_GLOBAL))

# show train-test distribution
df_train_test_dist = pd.DataFrame([df_train.label_text.value_counts().rename("train"), df_test.label_text.value_counts().rename("test"), df_cl.label_text.value_counts().rename("data_all")]).transpose()
df_train_test_dist




# show label distribution across languages
test_language = df_train.groupby(by="language", group_keys=True, as_index=True, sort=False).apply(lambda x: x.label_text.value_counts())
pd.DataFrame(test_language)




df_cl[df_cl.label_text != "no_topic"]




### test certainty of coders per category
import matplotlib.pyplot as plt

for name_df, group_df in df_cl.groupby(by="label_text", group_keys=False, as_index=False, sort=False):
  group_df.certainty_selection.value_counts(bins=5).plot.bar()
  print(name_df)
  plt.show()

# coders were relatively certain, less so for immigrant integration neutral (& sceptical)


