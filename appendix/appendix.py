# script for creating the data for the appendix

# load relevant packages
import pandas as pd
import numpy as np
import os

SEED_GLOBAL = 42



### load cleaned data
## PImPo
df_pimpo = pd.read_csv(f"./data-clean/df_pimpo_samp.zip")
df_pimpo_train = pd.read_csv(f"./data-clean/df_pimpo_samp_a1_train.zip")
df_pimpo_test = pd.read_csv(f"./data-clean/df_pimpo_samp_a1_test.zip")
# clean for immigration task
df_pimpo = df_pimpo[df_pimpo.label_text.isin(['immigration_neutral', 'immigration_sceptical', 'immigration_supportive', "no_topic"])]
df_pimpo_train = df_pimpo_train[df_pimpo_train.label_text.isin(['immigration_neutral', 'immigration_sceptical', 'immigration_supportive', "no_topic"])]
df_pimpo_test = df_pimpo_test[df_pimpo_test.label_text.isin(['immigration_neutral', 'immigration_sceptical', 'immigration_supportive', "no_topic"])]

## Manifesto-8
df_manifesto = pd.read_csv(f"./data-clean/df_manifesto_all.zip")
df_manifesto_train = pd.read_csv(f"./data-clean/df_manifesto-8_samp_train.zip")
df_manifesto_test = pd.read_csv(f"./data-clean/df_manifesto-8_samp_test.zip")
# adjust label_text column
df_manifesto["label_text"] = df_manifesto["label_domain_text"]
df_manifesto_train["label_text"] = df_manifesto_train["label_domain_text"]
df_manifesto_test["label_text"] = df_manifesto_test["label_domain_text"]

### Analysis 1:

## Manifesto-8
inspection_lang_manifesto_train_dic = {}
for lang in sorted(df_manifesto_train.language_iso.unique()):
    inspection_lang_manifesto_train_dic.update({lang: df_manifesto_train[df_manifesto_train.language_iso == lang].label_text.value_counts()})
df_inspection_manifesto_train_lang = pd.DataFrame(inspection_lang_manifesto_train_dic)
# test distribution
inspection_lang_manifesto_test_dic = {}
for lang in sorted(df_manifesto_test.language_iso.unique()):
    inspection_lang_manifesto_test_dic.update({lang: df_manifesto_test[df_manifesto_test.language_iso == lang].label_text.value_counts()})
df_inspection_manifesto_test_lang = pd.DataFrame(inspection_lang_manifesto_test_dic)

## PImPo
# train distribution
inspection_lang_pimpo_train_dic = {}
for lang in sorted(df_pimpo_train.language_iso.unique()):
    inspection_lang_pimpo_train_dic.update({lang: df_pimpo_train[df_pimpo_train.language_iso == lang].label_text.value_counts()})
df_inspection_pimpo_train_lang = pd.DataFrame(inspection_lang_pimpo_train_dic)
# test distribution
inspection_lang_pimpo_test_dic = {}
for lang in sorted(df_pimpo_test.language_iso.unique()):
    inspection_lang_pimpo_test_dic.update({lang: df_pimpo_test[df_pimpo_test.language_iso == lang].label_text.value_counts()})
df_inspection_pimpo_test_lang = pd.DataFrame(inspection_lang_pimpo_test_dic)


## Analysis 1 - write Excel files for appendix
df_inspection_manifesto_train_lang.to_excel("./appendix/df_manifesto_train_distribution_lang.xlsx", index=True)
df_inspection_manifesto_test_lang.to_excel("./appendix/df_manifesto_test_distribution_lang.xlsx", index=True)
df_inspection_pimpo_train_lang.to_excel("./appendix/df_pimpo_train_distribution_lang.xlsx", index=True)
df_inspection_pimpo_test_lang.to_excel("./appendix/df_pimpo_test_distribution_lang.xlsx", index=True)




### Analysis 2:

# downsampled no_topic for faster testing
SAMPLE_NO_TOPIC = 50_000
df_pimpo_samp = df_pimpo.groupby(by="label_text", as_index=False, group_keys=False).apply(lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_GLOBAL) if x.label_text.iloc[0] == "no_topic" else x)

## inspect label distributions
# language
inspection_a2_lang_dic = {}
for lang in df_pimpo_samp.language_iso.unique():
    inspection_a2_lang_dic.update({lang: df_pimpo_samp[df_pimpo_samp.language_iso == lang].label_text.value_counts()})
df_inspection_a2_lang = pd.DataFrame(inspection_a2_lang_dic)
# party family
inspection_a2_parfam_dic = {}
for parfam in df_pimpo_samp.parfam_text.unique():
    inspection_a2_parfam_dic.update({parfam: df_pimpo_samp[df_pimpo_samp.parfam_text == parfam].label_text.value_counts()})
df_inspection_a2_parfam = pd.DataFrame(inspection_a2_parfam_dic)
# country
inspection_a2_country_dic = {}
for country_iso in df_pimpo_samp.country_iso.unique():
    inspection_a2_country_dic.update({country_iso: df_pimpo_samp[df_pimpo_samp.country_iso == country_iso].label_text.value_counts()})
df_inspection_a2_country = pd.DataFrame(inspection_a2_country_dic)

# these tables still include the gaelic Irish texts for English and potential swedish texts as finnish
df_inspection_a2_lang.to_excel("./appendix/df_pimpo_a2_distribution_lang.xlsx", index=True)
df_inspection_a2_parfam.to_excel("./appendix/df_pimpo_a2_distribution_parfam.xlsx", index=True)
df_inspection_a2_country.to_excel("./appendix/df_pimpo_a2_distribution_country.xlsx", index=True)



