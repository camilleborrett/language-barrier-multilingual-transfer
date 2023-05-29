

# !!! delete API-key before publication !!!


# ## Install and load packages

import pandas as pd
import numpy as np
import os

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


# ## Load & prepare data

print(os.getcwd())
#os.chdir("./")
print(os.getcwd())


## load dfs
# correct v5 codebook: https://manifesto-project.wzb.eu/down/papers/handbook_2021_version_5.pdf - the PDF on the following website is wrong, but html is correct: https://manifesto-project.wzb.eu/coding_schemes/mp_v5
# we are working with v4 for backwards compatibility https://manifesto-project.wzb.eu/down/papers/handbook_2011_version_4.pdf
# overview of changes from v4 to v5: https://manifesto-project.wzb.eu/down/papers/Evolution_of_the_Manifesto_Coding_Instructions_and_the_Category_Scheme.pdf
# switch was 2016/2017
# ! adapt this downstream

#df = pd.read_csv("./data-raw/manifesto_all_2021a.csv", index_col="idx")
# write raw large .csv file to zip
#dataset_name = "manifesto_all_2021a"
#df.to_csv(f'./data-raw/{dataset_name}.zip', compression={"method": "zip", "archive_name": f"{dataset_name}.csv"})  # index=False

df = pd.read_csv("./data-raw/manifesto_all_2021a.zip", index_col="idx")
print(df.columns)
print(len(df))

df_cl = df.copy(deep=True)


#### explore languages
print(df_cl.country_name.value_counts())
print(df_cl.language.unique())
print(len(df_cl.language_iso.unique()))


## only work with selected languages
language_filter = ["en", "de", "es", "fr", "ko", "tr", "ru"]  # "ja"  excluding japanese because too little data. only 2 manifestos with 474 cleaned sents, only 7 of 8 domains
df_cl = df_cl[df_cl.language_iso.isin(language_filter)]
df_cl.rename(columns={"countryname": "country_name"}, inplace=True)


#### Data Cleaning
# check for NAs
print(len(df_cl))
df_cl = df_cl[~df_cl["text"].isna()]
print(len(df_cl))
# remove headlines
df_cl = df_cl[~df_cl["cmp_code"].isna()]  # 13k NA in English data. seem to be headlines and very short texts. they can have meaning
print(len(df_cl))
df_cl = df_cl[~df_cl["cmp_code"].str.match("H", na=False)]  # 7.6k headlines
print(len(df_cl))
# remove soviet codes
df_cl = df_cl[df_cl.type != "cee"]
df_cl = df_cl[~df_cl.title_variable.str.contains("Transition")]  # for some reason this is also hb5
print(len(df_cl))

# duplicates
# remove texts where exact same string has different code? Can keep it for experiments with context - shows value of context for disambiguation
#df_cl = df_cl.groupby(by="text").filter(lambda x: len(x.cmp_code.unique()) == 1)
#print(len(df_cl))
# maintain duplicates to maintain sequentiality of texts
#df_cl = df_cl[~df_cl.text.duplicated(keep="first")]  # around 7k
#print(len(df_cl))

# remove very short and long strings - too much noise
df_cl = df_cl[df_cl.text.str.len().ge(30)]
print(len(df_cl))
df_cl = df_cl[~df_cl.text.str.len().ge(1000)]  # remove very long descriptions, assuming that they contain too much noise from other types and unrelated language. 1000 characters removes around 9k
print(len(df_cl))


#### adapt variables to harmonised codebook version (v4 for backwards compatibility)
"""import requests
api_key = "f4aca6cabddc5170b7aaf41f8119af45"
corp_v = "MPDS2021a"  #"MPDS2020a"
meta_v = "2021-1"   #"2020-1"
# api_get_core_codebook
url = "https://manifesto-project.wzb.eu/tools/api_get_core_codebook.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
params = dict(api_key=api_key, key=corp_v)  # MPDS2020a
response = requests.get(url=url, params=params)
data_codebook = response.json()
df_codebook = pd.DataFrame(columns=data_codebook[0], data=data_codebook[1:])
df_codebook.domain_name = df_codebook.domain_name.replace("NA", "No other category applies")
df_codebook.to_csv("./data-preparation/codebook_categories_MPDS2021a-1.csv", index=False)
"""
df_codebook = pd.read_csv("./data-preparation/codebook_categories_MPDS2021a-1.csv")


### harmonise variable codes and titles
# translating label codes to label text with codebook mapping. MPDS2021a
# see codebook https://manifesto-project.wzb.eu/down/papers/handbook_2011_version_4.pdf
# Note that the "main" codes are from v4 for backwards compatibility with older data
# for new v5 categories: everything was aggregated up into the old v4 categories, except for 202.2, 605.2 und 703.2, which where added to 000.

df_cl = df_cl.rename(columns={"title_variable": "label_subcat_text", "domain_name": "label_domain_text"})  # give proper names to variables
df_cl.drop(columns=["domain_code", "code", "variable_name", "label", "type", "manual"], inplace=True)  # remove duplicate traces of different variable titles depending on codebook
#df_cl.cmp_code = ["000" if str(cmp_code) == "0.0" else str(cmp_code) if str(cmp_code)  else cmp_code for cmp_code in df_cl.cmp_code]  # for some reason some 0.0 values

## add selected codes to 0.0, because they contradict the hb4 coding:  202.2 (democracy general negative , 605.2 (law order negative) und 703.2 (agri negative)
print(len(df_cl.cmp_code.value_counts()))
cmp_code_other = ["202.2", "605.2", "703.2"]
df_cl.cmp_code = [str(cmp_code) if cmp_code not in cmp_code_other else "0.0" for cmp_code in df_cl.cmp_code]
#df_cl.label_subcat_text = [str(label_subcat_text) if cmp_code not in cmp_code_other else "No other category applies" for cmp_code, label_subcat_text in zip(df_cl.cmp_code, df_cl.label_subcat_text)]
#df_cl.cmp_code = [str(label) if label not in cmp_code_other else "0.0" for label in df_cl.cmp_code]
print(len(df_cl.cmp_code.value_counts()))

## harmonise subcat variable titles to v4 codebook
# a code with XX.1 or XX.2 indicates that its codebook v5. reduce all of them to XX.0
df_cl["cmp_code_v4"] = [str(label) + ".0" if str(label)[-2:] not in [".0", ".1", ".2", ".3", ".4"] else str(label)[:-2] + ".0" for label in df_cl.cmp_code]
df_cl["cmp_code_v4"] = ["0.0" if label == "000.0" else label for label in df_cl.cmp_code_v4]
print(len(df_cl["cmp_code_v4"].value_counts()))
# then get the respective variable text for all XXX.0 variables
cmp_code_to_variable_title_map = {str(cmp_code): str(variable_title) for cmp_code, variable_title in zip(df_codebook.code, df_codebook.title)}
df_cl.label_subcat_text = [cmp_code_to_variable_title_map[cmp_code_v4] for cmp_code_v4 in df_cl.cmp_code_v4]
print(len(df_cl.label_subcat_text.value_counts()))

## harmonise domain variable titles
print(len(df_cl.label_domain_text.value_counts()))
df_cl["label_domain_text"] = [domain_text if not pd.isna(domain_text) else "No other category applies" for domain_text in df_cl.label_domain_text]
print(len(df_cl.label_domain_text.value_counts()))

## tests
assert len(df_cl.cmp_code_v4.value_counts()) == 57
assert len(df_cl.label_subcat_text.value_counts()) == 57
assert len(df_cl.label_domain_text.value_counts()) == 8




## decide on label level to use for downstream analysis
#df_cl["label_text"] = df_cl["label_domain_text"]
#df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]
# test that label and label_text correspond
#assert len(df_cl[df_cl.label_text.isna()]) == 0  # each label_cap2 could be mapped to a label text. no label text is missing.
#assert all(np.sort(df_cl["label_text"].value_counts().tolist()) == np.sort(df_cl["label"].value_counts().tolist()))

# final update
df_cl = df_cl.reset_index(drop=True)
df_cl.index = df_cl.index.rename("idx")  # name index. provides proper column name in dataset object downstream

print(df_cl.label_domain_text.value_counts(), "\n")
print(df_cl.country_name.value_counts())
print(df_cl.language_iso.value_counts())





### augmenting text column

## new column where every sentence is merged with previous sentence
n_unique_doc_lst = []
n_unique_doc = 0
text_preceding = []
text_following = []
for name_group, df_group in df_cl.groupby(by=["manifesto_id"], sort=False):  # over each speech to avoid merging sentences accross manifestos
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
          raise Exception("Issue: condition not in code")
    n_unique_doc_lst.append([n_unique_doc] * len(df_group["text"]))
n_unique_doc_lst = [item for sublist in n_unique_doc_lst for item in sublist]

# create new columns
df_cl["text_original"] = df_cl["text"]
df_cl = df_cl.drop(columns=["text"])
df_cl["text_preceding"] = text_preceding
df_cl["text_following"] = text_following
df_cl["doc_id"] = n_unique_doc_lst  # column with unique doc identifier



## remove and reorder columns
df_cl.columns
df_cl = df_cl[["language_iso", "text_original", "text_preceding", "text_following",  #"label", "label_text",
               "manifesto_id", "sentence_id", "doc_id", "country_iso", "date", "party", "partyname", "parfam",
               "cmp_code_v4", "label_domain_text", "label_subcat_text"]]





##### Train-Test-Split
from sklearn.model_selection import train_test_split

## stratify by two variables: lang & subcat_text  # https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
df_cl['stratify_by'] = df_cl['language_iso'].astype(str) + "_" + df_cl['label_subcat_text'].astype(str)
# delete sub-cat for languages when it has only very few examples. otherwise testing with train-test split is not possible
label_subcat_lang_count = df_cl.groupby(by="language_iso").apply(lambda x: x.label_subcat_text.value_counts())
min_number_subcat_texts = 2
low_n_lang_cat = [lang_label_tuple[0] + "_" + lang_label_tuple[1] for lang_label_tuple in label_subcat_lang_count[label_subcat_lang_count < min_number_subcat_texts].index]
print(len(df_cl))
df_cl = df_cl[~df_cl.stratify_by.isin(low_n_lang_cat)]
print(len(df_cl))
# stratified sample
df_train, df_test = train_test_split(df_cl, test_size=0.30, random_state=SEED_GLOBAL, stratify=df_cl["stratify_by"])

print(f"Overall train size: {len(df_train)}")
#print(f"Overall test size: {len(df_test)} - sampled test size: {len(df_test_samp)}")
print(f"Overall test size: {len(df_test)}")

## train-test distribution
df_train_test_distribution = pd.DataFrame([df_train.label_domain_text.value_counts().rename("train"), df_test.label_domain_text.value_counts().rename("test"),
                                           #df_test_samp.label_domain_text.value_counts().rename("test_sample"),
                                           df_cl.label_domain_text.value_counts().rename("all")]).transpose()
## label distribution by language
df_distribution_lang_domain = []
for key_language, value_df in df_cl.groupby(by="language_iso", group_keys=True, as_index=True, sort=False, axis=0):
    df_distribution_lang_domain.append(value_df.label_domain_text.value_counts(normalize=False).round(2).rename(key_language))
df_distribution_lang_domain = pd.concat(df_distribution_lang_domain, axis=1)

df_distribution_lang_subcat = []
for key_language, value_df in df_cl.groupby(by="language_iso", group_keys=True, as_index=True, sort=False, axis=0):
    df_distribution_lang_subcat.append(value_df.label_subcat_text.value_counts(normalize=False).round(2).rename(key_language))
df_distribution_lang_subcat = pd.concat(df_distribution_lang_subcat, axis=1)


#### manifesto-8 sample
## take sample - do not need all data, since only using sample size of 1k (maybe 10k) to reduce excessive compute
# train
# at least 3 for each subcat to avoid algo issues downstream
df_train_samp_min_subcat = df_train.groupby(by="language_iso").apply(lambda x: x.groupby(by="label_subcat_text").apply(lambda x: x.sample(n=min(len(x), 3), random_state=42)))
df_train_samp = df_train.groupby(by="language_iso").apply(lambda x: x.sample(n=min(len(x), 5_000), random_state=42))
df_train_samp = pd.concat([df_train_samp, df_train_samp_min_subcat])
df_train_samp = df_train_samp[~df_train_samp.text_original.duplicated(keep='first')]
df_train_samp = df_train_samp.reset_index(drop=True)
print(len(df_train_samp))
print(df_train_samp.language_iso.value_counts())
# test
#df_test = df_test.groupby(by="language_iso").apply(lambda x: x.groupby(by="label_subcat_text").apply(lambda x: x.sample(n=min(len(x), 50), random_state=42)))
df_test_samp_min_subcat = df_test.groupby(by="language_iso").apply(lambda x: x.groupby(by="label_subcat_text").apply(lambda x: x.sample(n=min(len(x), 3), random_state=42)))
df_test_samp = df_test.groupby(by="language_iso").apply(lambda x: x.sample(n=min(len(x), 4_000), random_state=42))
df_test_samp = pd.concat([df_test_samp, df_test_samp_min_subcat])
df_test_samp = df_test_samp[~df_test_samp.text_original.duplicated(keep='first')]
df_test_samp = df_test_samp.reset_index(drop=True)
print(len(df_test_samp))
print(df_test_samp.language_iso.value_counts())



#### Save data
print(os.getcwd())

### write
#df_cl.to_csv("./data-clean/df_manifesto_all.csv")
#df_train.to_csv("./data-clean/df_manifesto-8_train.csv")
#df_test.to_csv("./data-clean/df_manifesto-8_test.csv")

name_df_all = "df_manifesto_all"
name_df_train = "df_manifesto-8_samp_train"
name_df_test = "df_manifesto-8_samp_test"
#compression_options = dict(method='zip', archive_name=f'....csv')
df_cl.to_csv(f'./data-clean/{name_df_all}.zip', compression={"method": "zip", "archive_name": f"{name_df_all}.csv"}, index=False)
df_train_samp.to_csv(f'./data-clean/{name_df_train}.zip', compression={"method": "zip", "archive_name": f"{name_df_train}.csv"}, index=False)
df_test_samp.to_csv(f'./data-clean/{name_df_test}.zip', compression={"method": "zip", "archive_name": f"{name_df_test}.csv"}, index=False)


"""
df_cl_military.to_csv("./data-clean/df_manifesto_military_cl.csv")
df_train_military.to_csv("./data-clean/df_manifesto_military_train.csv")
df_test_military.to_csv("./data-clean/df_manifesto_military_test.csv")

df_cl_protectionism.to_csv("./data-clean/df_manifesto_protectionism_cl.csv")
df_train_protectionism.to_csv("./data-clean/df_manifesto_protectionism_train.csv")
df_test_protectionism.to_csv("./data-clean/df_manifesto_protectionism_test.csv")

df_cl_morality.to_csv("./data-clean/df_manifesto_morality_cl.csv")
df_train_morality.to_csv("./data-clean/df_manifesto_morality_train.csv")
df_test_morality.to_csv("./data-clean/df_manifesto_morality_test.csv")
"""








#### old code

## sample based on docs - to make test set composed of entirely different docs - avoid data leakage when including surrounding sentences
# not used because: causes downstream issues for very small languages
#df_cl.groupby(by="language_iso").apply(lambda x: len(x.label_domain_text))
"""doc_id_train = pd.Series(df_cl.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test = df_cl[~df_cl.doc_id.isin(doc_id_train)].doc_id.unique().tolist()
print(len(doc_id_train))
print(len(doc_id_test))
assert sum([doc_id in doc_id_train for doc_id in doc_id_test]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train = df_cl[df_cl.doc_id.isin(doc_id_train)]
df_test = df_cl[~df_cl.doc_id.isin(doc_id_train)]"""


# ### Small complex tasks
"""
### Military positive vs. negative

df_cl_military = df_cl.copy(deep=True)

label_text_military = [label if label in ["Military: Positive", "Military: Negative"] else "Other" for label in df_cl_military.label_subcat_text]

df_cl_military["label_text"] = label_text_military
df_cl_military["label"] = pd.factorize(df_cl_military["label_text"], sort=True)[0]

## train test split
# simple split
#df_train_military, df_test_military = train_test_split(df_cl_military, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl_military["label_text"])
# better train test split to avoid data leakage
doc_id_train_military = pd.Series(df_cl_military.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_military = df_cl_military[~df_cl_military.doc_id.isin(doc_id_train_military)].doc_id.unique().tolist()
print(len(doc_id_train_military))
print(len(doc_id_test_military))
assert sum([doc_id in doc_id_train_military for doc_id in doc_id_test_military]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_military = df_cl_military[df_cl_military.doc_id.isin(doc_id_train_military)]
df_test_military = df_cl_military[~df_cl_military.doc_id.isin(doc_id_train_military)]

# down sampling the "other" category
df_train_military = df_train_military.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train_military.label_text != "Other")*1), random_state=SEED_GLOBAL))
df_test_military = df_test_military.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test_military.label_text != "Other")*10), random_state=SEED_GLOBAL))
df_cl_military = pd.concat([df_train_military, df_test_military])

# show data distribution
print(f"Overall train size: {len(df_train_military)}")
print(f"Overall test size: {len(df_test_military)}")
df_train_test_distribution_military = pd.DataFrame([df_train_military.label_text.value_counts().rename("train"), df_test_military.label_text.value_counts().rename("test"),
                                                   df_cl_military.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_military

## label distribution by language
df_distribution_lang_military = []
for key_language, value_df in df_cl_military.groupby(by="language_iso", group_keys=True, as_index=True, sort=False, axis=0):
    df_distribution_lang_military.append(value_df.label_text.value_counts(normalize=True).round(2).rename(key_language))
df_distribution_lang_military = pd.concat(df_distribution_lang_military, axis=1)


### Protectionism positive vs. negative
df_cl_protectionism = df_cl.copy(deep=True)

label_text_protectionism = [label if label in ["Protectionism: Positive", "Protectionism: Negative"] else "Other" for label in df_cl_protectionism.label_subcat_text]

df_cl_protectionism["label_text"] = label_text_protectionism
df_cl_protectionism["label"] = pd.factorize(df_cl_protectionism["label_text"], sort=True)[0]

## train test split
# simple split
#df_train_protectionism, df_test_protectionism = train_test_split(df_cl_protectionism, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl_protectionism["label_text"])
# better train test split to avoid data leakage
doc_id_train_protectionism = pd.Series(df_cl_protectionism.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_protectionism = df_cl_protectionism[~df_cl_protectionism.doc_id.isin(doc_id_train_protectionism)].doc_id.unique().tolist()
print(len(doc_id_train_protectionism))
print(len(doc_id_test_protectionism))
assert sum([doc_id in doc_id_train_protectionism for doc_id in doc_id_test_protectionism]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_protectionism = df_cl_protectionism[df_cl_protectionism.doc_id.isin(doc_id_train_protectionism)]
df_test_protectionism = df_cl_protectionism[~df_cl_protectionism.doc_id.isin(doc_id_train_protectionism)]

# down sampling the "other" category
df_train_protectionism = df_train_protectionism.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train_protectionism.label_text != "Other")*1), random_state=SEED_GLOBAL))
df_test_protectionism = df_test_protectionism.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test_protectionism.label_text != "Other")*10), random_state=SEED_GLOBAL))
df_cl_protectionism = pd.concat([df_train_protectionism, df_test_protectionism])

# show data distribution
print(f"Overall train size: {len(df_train_protectionism)}")
print(f"Overall test size: {len(df_test_protectionism)}")
df_train_test_distribution_protectionism = pd.DataFrame([df_train_protectionism.label_text.value_counts().rename("train"), df_test_protectionism.label_text.value_counts().rename("test"),
                                                   df_cl_protectionism.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_protectionism

## label distribution by language
df_distribution_lang_protectionism = []
for key_language, value_df in df_cl_protectionism.groupby(by="language_iso", group_keys=True, as_index=True, sort=False, axis=0):
    df_distribution_lang_protectionism.append(value_df.label_text.value_counts(normalize=True).round(2).rename(key_language))
df_distribution_lang_protectionism = pd.concat(df_distribution_lang_protectionism, axis=1)


### Traditional Morality positive vs. negative

df_cl_morality = df_cl.copy(deep=True)

label_text_morality = [label if label in ["Traditional Morality: Positive", "Traditional Morality: Negative"] else "Other" for label in df_cl_morality.label_subcat_text]

df_cl_morality["label_text"] = label_text_morality
df_cl_morality["label"] = pd.factorize(df_cl_morality["label_text"], sort=True)[0]

## train test split
# simple split
#df_train_morality, df_test_morality = train_test_split(df_cl_morality, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl_morality["label_text"])
# better train test split to avoid data leakage
doc_id_train_morality = pd.Series(df_cl_morality.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_morality = df_cl_morality[~df_cl_morality.doc_id.isin(doc_id_train_morality)].doc_id.unique().tolist()
print(len(doc_id_train_morality))
print(len(doc_id_test_morality))
assert sum([doc_id in doc_id_train_morality for doc_id in doc_id_test_morality]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_morality = df_cl_morality[df_cl_morality.doc_id.isin(doc_id_train_morality)]
df_test_morality = df_cl_morality[~df_cl_morality.doc_id.isin(doc_id_train_morality)]

# down sampling the "other" category
df_train_morality = df_train_morality.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train_morality.label_text != "Other")*1), random_state=SEED_GLOBAL))
df_test_morality = df_test_morality.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test_morality.label_text != "Other")*10), random_state=SEED_GLOBAL))
df_cl_morality = pd.concat([df_train_morality, df_test_morality])

# show data distribution
print(f"Overall train size: {len(df_train_morality)}")
print(f"Overall test size: {len(df_test_morality)}")
df_train_test_distribution_morality = pd.DataFrame([df_train_morality.label_text.value_counts().rename("train"), df_test_morality.label_text.value_counts().rename("test"),
                                                   df_cl_morality.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_morality

## label distribution by language
df_distribution_lang_morality = []
for key_language, value_df in df_cl_morality.groupby(by="language_iso", group_keys=True, as_index=True, sort=False, axis=0):
    df_distribution_lang_morality.append(value_df.label_text.value_counts(normalize=True).round(2).rename(key_language))
df_distribution_lang_morality = pd.concat(df_distribution_lang_morality, axis=1)


### National Way of Life positive vs. negative

df_cl_nationalway = df_cl.copy(deep=True)

label_text_nationalway = [label if label in ["National Way of Life: Positive", "National Way of Life: Negative"] else "Other" for label in df_cl_nationalway.label_subcat_text]

df_cl_nationalway["label_text"] = label_text_nationalway
df_cl_nationalway["label"] = pd.factorize(df_cl_nationalway["label_text"], sort=True)[0]

## train test split
# simple split
#df_train_nationalway, df_test_nationalway = train_test_split(df_cl_nationalway, test_size=0.25, random_state=SEED_GLOBAL, stratify=df_cl_nationalway["label_text"])
# better train test split to avoid data leakage
doc_id_train_nationalway = pd.Series(df_cl_nationalway.doc_id.unique()).sample(frac=0.70, random_state=SEED_GLOBAL).tolist()
doc_id_test_nationalway = df_cl_nationalway[~df_cl_nationalway.doc_id.isin(doc_id_train_nationalway)].doc_id.unique().tolist()
print(len(doc_id_train_nationalway))
print(len(doc_id_test_nationalway))
assert sum([doc_id in doc_id_train_nationalway for doc_id in doc_id_test_nationalway]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train_nationalway = df_cl_nationalway[df_cl_nationalway.doc_id.isin(doc_id_train_nationalway)]
df_test_nationalway = df_cl_nationalway[~df_cl_nationalway.doc_id.isin(doc_id_train_nationalway)]

# down sampling the "other" category
df_train_nationalway = df_train_nationalway.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_train_nationalway.label_text != "Other")*1), random_state=SEED_GLOBAL))
df_test_nationalway = df_test_nationalway.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), sum(df_test_nationalway.label_text != "Other")*10), random_state=SEED_GLOBAL))
df_cl_nationalway = pd.concat([df_train_nationalway, df_test_nationalway])

# show data distribution
print(f"Overall train size: {len(df_train_nationalway)}")
print(f"Overall test size: {len(df_test_nationalway)}")
df_train_test_distribution_nationalway = pd.DataFrame([df_train_nationalway.label_text.value_counts().rename("train"), df_test_nationalway.label_text.value_counts().rename("test"),
                                                       df_cl_nationalway.label_text.value_counts().rename("all")]).transpose()
df_train_test_distribution_nationalway

## label distribution by language
df_distribution_lang_nationalway = []
for key_language, value_df in df_cl_nationalway.groupby(by="language_iso", group_keys=True, as_index=True, sort=False, axis=0):
    df_distribution_lang_nationalway.append(value_df.label_text.value_counts(normalize=True).round(2).rename(key_language))
df_distribution_lang_nationalway = pd.concat(df_distribution_lang_nationalway, axis=1)

"""





## test how many sentences have same type as preceding / following sentence
"""test_lst = []
test_lst2 = []
test_lst_after = []
for name_df, group_df in df_cl.groupby(by="doc_id", group_keys=False, as_index=False, sort=False):
  for i in range(len(group_df)):
    # one preceding text
    if i == 0:
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i-1]:
      test_lst.append("same_before")
    else:
      #test_lst.append(f"different label before: {group_df['label_text'].iloc[i-1]}")
      test_lst.append(f"different label before")
    # for following texts
    if i >= len(group_df)-1:
      continue
    elif group_df["label_text"].iloc[i] == group_df["label_text"].iloc[i+1]:
      test_lst_after.append("same_after")
    else:
      #test_lst_after.append(f"different label after: {group_df['label_text'].iloc[i+1]}")
      test_lst_after.append(f"different label after")

print(pd.Series(test_lst).value_counts(normalize=True), "\n")
print(pd.Series(test_lst_after).value_counts(normalize=True), "\n")"""
# SOTU: 75 % of sentences have the same type as the preceeding sentence. also 75% for following sentence. #  concatenating preceding/following leads to data leakage? 25% different class which can confuse the model, its's random and same for all models
# Manifesto: 57 % of sentences have same type as preceding sentence (57 class). # including preceding sentence should not provide illegitimate advantage to classifier


