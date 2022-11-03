#!/usr/bin/env python
# coding: utf-8

# !!! delete API-key before publication !!!


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



## load data
# PImPo codebook: https://manifesto-project.wzb.eu/down/datasets/pimpo/PImPo_codebook.pdf
df = pd.read_csv("https://raw.githubusercontent.com/farzamfan/Multilingual-StD-Pipeline/main/Plm%20Manifest/PImPo%20dataset/PImPo_verbatim.csv", sep=",",  #encoding='utf-8',  # low_memory=False  #lineterminator='\t',
                 on_bad_lines='skip'
)
print(df.columns)
print(len(df))



### data cleaning
# codebook https://manifesto-project.wzb.eu/down/datasets/pimpo/PImPo_codebook.pdf
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
# !! should me this iso2-char
# languages used in manifesto-8: ["en", "de", "es", "fr", "ko", "tr", "ru"]
"""languages_map = {
    11: "swe", 12: "nor", 13: "dan", 14: "fin", 22: "nld",
    33: "spa", 41: "deu", 42: "deu", 43: "deu",  # manually checked Switzerland, only has deu texts
    53: "eng", 61: "eng", 62: "fra", 63: "eng", 64: "eng"  # manually checked Canada, only has fra texts
}"""
languages_map = {
    11: "sv", 12: "nn", 13: "da", 14: "fi", 22: "nl",
    33: "es", 41: "de", 42: "de", 43: "de",  # manually checked Switzerland, only has deu texts
    53: "en", 61: "en", 62: "fr", 63: "en", 64: "en"  # manually checked Canada, only has fra texts
}
df_cl["language_iso"] = [languages_map[country_id] for country_id in df_cl.country]

print(df_cl.language_iso.value_counts())
print(df_cl.language_iso.unique())

## select languages to analyse
# available languages: ['sv' 'nn' 'da' 'fi' 'nl' 'es' 'de' 'en' 'fr']
# reasons for selection: spoken languages by team; pairs of related langs; unrelated langs; n data points in dataset
#df_cl = df_cl[df_cl.language_iso.isin(["en", "de", "nl", "fr", "es", "fi", "nn"])]
print(df_cl.language_iso.value_counts())



#### add party meta data via MARPOR
## Get core dataset - one manifesto per row - aggregated values, no sentences
import requests
# Parameters
api_key = "f4aca6cabddc5170b7aaf41f8119af45"
corp_v = "MPDS2021a"  #"MPDS2020a"
meta_v = "2021-1"   #"2020-1"
# download
url = "https://manifesto-project.wzb.eu/tools/api_get_core.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
params = dict(api_key=api_key, key=corp_v)
response = requests.get(url=url, params=params)
data_core = response.json()  # JSON Response Content documentation: https://requests.readthedocs.io/en/master/user/quickstart/#json-response-content
# create df of core manifesto dataset
df_core = pd.DataFrame(columns=data_core[0], data=data_core[1:])

df_core = df_core[["party", "partyname", "partyabbrev", "parfam"]]
df_core["party"] = df_core["party"].astype(int)
df_core = df_core[~df_core["party"].duplicated(keep='first')] # only one row per party for unique meta data
df_cl["party"] = df_cl["party"].astype(int)

# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
df_cl = pd.merge(df_cl, df_core, left_on="party", right_on="party", how="left")

# party family text mapping https://manifesto-project.wzb.eu/down/data/2020a/codebooks/codebook_MPDataset_MPDS2020a.pdf
parfam_map = {10: "ECO", 20: "LEF", 30: "SOC", 40: "LIB", 50: "CHR", 60: "CON", 70: "NAT",
              80: "AGR", 90: "ETH", 95: "SIP", 98: "DIV", 999: "MI"}  # not used in pimpo paper: SIP, DIV, MI
df_cl["parfam"] = df_cl["parfam"].astype(int)
df_cl["parfam_text"] = df_cl.parfam.map(parfam_map)


## ! delete parfams not used in pimpo paper
parfam_paper_lst = ["ECO", "ETH", "AGR", "LEF", "CHR", "LIB", "SOC", "CON", "NAT"]
df_cl = df_cl[df_cl.parfam_text.isin(parfam_paper_lst)]

df_cl["parfam_text"].value_counts()





### key variables:
# df_cl.selection: "Variable takes the value of 1 if immigration or immigrant integration related, and zero otherwise."
# df_cl.topic: 1 == immigration, 2 == integration, NaN == not related
# df_cl.direction: -1 == sceptical, 0 == neutral, 1 == supportive, NaN == not related to topics. "Gives the direction of a quasi-sentences as either sceptical, neutral or supportive. Missing for quasi-sentence which are classified as not-related to immigration or integration."

### gold answer variables (used to test crowd coders)
## df_cl.gs_answer_1r == gold answers whether related to immigration or integration  used for testing crowd coders
# 0 == not immi/inti related; 1 == Immi/inti related  # "Comment: The variable gives the value of how the respective gold-sentence was coded by the authors, i.e. 0 if it was classified as not related to immigration or integration and 1 if it was regarded as related to one of these. It is missing if a quasi-sentence was not a gold sentence in the first round."
## df_cl.gs_answer_2q == Gold answers whether it is about immigration or integration or none of the two
# 1 == immigration, 2 == integration, NaN == none of the two
## df_cl.gs_answer_3q == gold answers supportive or sceptical or neutral towards topic
# -1 == sceptical, 1 == supportive, NaN == none of the two (probably neutral if also about topic)
#df_cl_gold = df_cl[['gs_answer_1r', 'gs_answer_2q', 'gs_answer_3q']]
#df_cl = df_cl[['selection', 'topic', 'direction', "text_preceding", 'text_original', "text_following"]]

df_cl = df_cl.reset_index(drop=True)

## add gold label text column
# interesting notes on quality and consistency:
# gold answer can diverge from crowd answer (e.g. index 232735).
# also: if sth was gold answer for r1, it's not necessarily gold answer for r2. no gold answer for topic neutral was provided, those three where only gold for r1 (!! one of them even has divergence between gold and crowd, index 232735)

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

# set final label and label_text column
df_cl["label_text"] = label_text_gold_lst
df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]

print(df_cl["label_text"].value_counts())




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
df_cl = df_cl[["label", "label_text", 'country_iso', 'language_iso', "doc_id", # 'gs_1r', 'gs_2r', 'date', 'party', 'pos_corpus'
               #'gs_answer_1r', 'gs_answer_2q', 'gs_answer_3q', # 'num_codings_1r', 'num_codings_2r'
              'selection', 'certainty_selection', 'topic', 'certainty_topic',
              'direction', 'certainty_direction', 'rn', 'cmp_code',  # 'manually_coded'
              'text_original', "text_preceding", "text_following",
               "partyname", "partyabbrev", "parfam", "parfam_text"
               ]]


## inspect label distributions
# language
inspection_lang_dic = {}
for lang in df_cl.language_iso.unique():
    inspection_lang_dic.update({lang: df_cl[df_cl.language_iso == lang].label_text.value_counts()})
df_inspection_lang = pd.DataFrame(inspection_lang_dic)
# party family
inspection_parfam_dic = {}
for parfam in df_cl.parfam_text.unique():
    inspection_parfam_dic.update({parfam: df_cl[df_cl.parfam_text == parfam].label_text.value_counts()})
df_inspection_parfam = pd.DataFrame(inspection_parfam_dic)


df_cl.to_csv("./data-clean/df_pimpo_all.csv", index=False)












### train-test-split
# ! probably don't need, because will try different split scenarios in analysis script
# better train test split to avoid data leakage
# ! copied code - double check and take design decisions
"""doc_id_train = pd.Series(df_cl.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test = df_cl[~df_cl.doc_id.isin(doc_id_train)].doc_id.unique().tolist()
print(len(doc_id_train))
print(len(doc_id_test))
assert sum([doc_id in doc_id_train for doc_id in doc_id_test]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train = df_cl[df_cl.doc_id.isin(doc_id_train)]
df_test = df_cl[~df_cl.doc_id.isin(doc_id_train)]

# down sampling the "other" category
# ! should probably do this by lang
df_train = df_train.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train.label_text != "no_topic")*1), random_state=SEED_GLOBAL))
df_test = df_test.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test.label_text != "no_topic")*10), random_state=SEED_GLOBAL))
df_cl = pd.concat([df_train, df_test])

# show data distribution
print(f"Overall train size: {len(df_train)}")
print(f"Overall test size: {len(df_test)}")
df_train_test_distribution = pd.DataFrame([df_train.label_text.value_counts().rename("train"), df_test.label_text.value_counts().rename("test"),
                                                   df_cl.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution"""


## test how many sentences have same type as preceding / following sentence
#df_test = df_cl[df_cl.label_text != "no_topic"]
"""
test_lst = []
test_lst2 = []
test_lst_after = []
for name_df, group_df in df_cl[df_cl.language_iso == "en"].groupby(by="doc_id", group_keys=False, as_index=False, sort=False):
  for i in range(len(group_df)):
    # one preceding text
    if i == 0 or group_df["label_text"].iloc[i] == "no_topic":
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1]:
      test_lst.append("same_before")
    else:
      test_lst.append(f"different label before: {group_df['label_text'].iloc[i-1]}")
    # two preceding texts
    #if i < 2 or group_df["label_text"].iloc[i] == "no_topic":
    #  continue
    #elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1] == group_df["label_text"].iloc[i-2]:
    #  test_lst2.append("same_two_before")
    #else:
    #  test_lst2.append("different_two_before")
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
"""

### test certainty of coders per category
"""import matplotlib.pyplot as plt
for name_df, group_df in df_cl.groupby(by="label_text", group_keys=False, as_index=False, sort=False):
  group_df.certainty_selection.value_counts(bins=5).plot.bar()
  print(name_df)
  plt.show()"""
# coders were relatively certain, less so for immigrant integration neutral (& sceptical)

