

# !!! Remove API key before publication !!!


# API documentation: https://manifesto-project.wzb.eu/information/documents/api
# my api key: f4aca6cabddc5170b7aaf41f8119af45

import pandas as pd
import requests
import re
import time
import os

### Parameters
api_key = "f4aca6cabddc5170b7aaf41f8119af45"
corp_v = "MPDS2021a"  #"MPDS2020a"
meta_v = "2021-1"   #"2020-1"

## check latest corpus and meta data versions
# api_list_core_versions
url = "https://manifesto-project.wzb.eu/tools/api_list_core_versions.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
params = dict(api_key=api_key)
response = requests.get(url=url, params=params)
data_core_versions = response.json()
# latest: {'id': 'MPDS2021a', 'name': 'Manifesto Project Dataset (version 2021a)'}

# api_list_metadata_versions
url = "https://manifesto-project.wzb.eu/tools/api_list_metadata_versions.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
params = dict(api_key=api_key, tag="true", details="true")
response = requests.get(url=url, params=params)
data_meta_versions = response.json()
# latest: {'name': '20211021091322', 'tag': '2021-1', 'description': 'to be written (added 2021a update, ...)'}




##### 1. Get core dataset - one manifesto per row - aggregated values, no sentences
url = "https://manifesto-project.wzb.eu/tools/api_get_core.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
params = dict(api_key=api_key, key=corp_v)
response = requests.get(url=url, params=params)
data_core = response.json()  # JSON Response Content documentation: https://requests.readthedocs.io/en/master/user/quickstart/#json-response-content
# create df of core manifesto dataset
df_core = pd.DataFrame(columns=data_core[0], data=data_core[1:])


##### 2. Get additional meta data per manifesto
# this contains information "annotations": "True"/"False"
#url = f"https://manifesto-project.wzb.eu/tools/api_metadata.json?api_key={api_key}&keys[]=41320_200909&keys[]=41320_200509&version=2015-3"

## get keys for each manifesto for api metadata
manifesto_keys = df_core['party'] + "_" + df_core["date"]
manifesto_keys = tuple([key for key in manifesto_keys])
# chunk manifestos to avoid server/API error  https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunk_lst(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]  # create chunks of size n
manifesto_keys_chunks = chunk_lst(manifesto_keys, 250)

## get meta data
def get_metadata(manifesto_keys, api_key, meta_v):
    url = "https://manifesto-project.wzb.eu/tools/api_metadata.json"  #?api_key=f4aca6cabddc5170b7aaf41f8119af45&keys=41320_200909&version=MPDS2020a"
    params = {"api_key": api_key, "keys[]": manifesto_keys, "version": meta_v}   # "keys[]": ("41320_200909", "41320_200509"),
    response = requests.get(url=url, params=params)
    data_manif_meta = response.json()
    time.sleep(3)
    return data_manif_meta

## download manifesto metadata in chunks to avoid error
data_manif_meta_chunks = [get_metadata(chunk, api_key, meta_v) for chunk in manifesto_keys_chunks]
# transform to df
data_manif_meta_lst = [data_manif_meta["items"] for data_manif_meta in data_manif_meta_chunks]
data_manif_meta_lst = [item for sublist in data_manif_meta_lst for item in sublist]
df_manif_meta = pd.DataFrame(data_manif_meta_lst)
# subset for manifestos only with annotations
df_manif_meta.columns
df_manif_meta_cl = df_manif_meta[df_manif_meta["annotations"] == True]

df_manif_meta_cl.language.value_counts()


##### 3. download full-text with annotations on sentence level
## api_texts_and_annotations

# if desired, filter only specific language of interest
#language_filter = ".*"
#df_manif_meta_cl = df_manif_meta_cl[df_manif_meta_cl["language"].str.contains(language_filter)]

## download
def get_texts(manifesto_ids, api_key, meta_v):
    url = "https://manifesto-project.wzb.eu/tools/api_texts_and_annotations.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
    params = {"api_key": api_key, "keys[]": manifesto_ids, "version": meta_v}  # "keys[]": "41320_200909",
    response = requests.get(url=url, params=params)
    data_text = response.json()
    time.sleep(3)
    return data_text

# download manifestos grouped by language:
data_text_dic = {}
for key_language, df_group in df_manif_meta_cl.groupby(by="language"):
    data_text = get_texts(df_group["manifesto_id"], api_key, meta_v)
    data_text = {key_language: data_text}
    data_text_dic.update(data_text)

## if not grouped by language - download manifestos in chunks
# chunk ids to avoid server/API error
#manifesto_ids = df_manif_meta_cl["manifesto_id"]
#def chunk_lst(l, n):
#    return [l[i:i+n] for i in range(0, len(l), n)]  # create chunks of size n
#manifesto_ids_chunks = chunk_lst(manifesto_ids, 250)
#data_text_lst = [get_texts(manifesto_ids) for manifesto_ids in manifesto_ids_chunks]

# create overall df with all annotated sentences for several manifestos
def create_df_text(data_text):
    df_text_lst = []
    for i in range(len(data_text["items"])):  # iterate over each manifesto
        # create df for each manifesto
        df_text = pd.DataFrame(data_text["items"][i]["items"])
        # rename first text column consistently to "text" (it's sometimes "content", sometimes "text")
        df_text.rename(columns={df_text.columns[0]: "text"}, inplace=True)
        # add other columns
        df_text["kind"] = [data_text["items"][i]["kind"]] * len(data_text["items"][i]["items"])
        df_text["key"] = [data_text["items"][i]["key"]] * len(data_text["items"][i]["items"])
        party_id = re.sub("_.*", "", data_text["items"][i]["key"])  # reconstruct party id from key
        df_text["party"] = [party_id] * len(data_text["items"][i]["items"])
        date = re.sub(".*_", "", data_text["items"][i]["key"])  # reconstruct date from key
        df_text["date"] = [date] * len(data_text["items"][i]["items"])
        df_text_lst.append(df_text)
    df_text = pd.concat(df_text_lst)
    return df_text

df_text_dic = {key_language: create_df_text(value_data_text) for key_language, value_data_text in data_text_dic.items()}
df_text = pd.concat(df_text_dic)



#### get codebook df with variables/codes
# api_get_core_codebook
url = "https://manifesto-project.wzb.eu/tools/api_get_core_codebook.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
params = dict(api_key=api_key, key=corp_v)  # MPDS2020a
response = requests.get(url=url, params=params)
data_codebook = response.json()
df_codebook = pd.DataFrame(columns=data_codebook[0], data=data_codebook[1:])
# clean codebook:
df_codebook_cl = df_codebook.copy()
df_codebook_cl["description_md"] = df_codebook_cl["description_md"].str.replace("\n", " ")
df_codebook_cl["description_md"] = df_codebook_cl["description_md"].str.replace(r"-\s+", "")
df_codebook_cl["description_md"] = df_codebook_cl["description_md"].str.replace(r"\s+", " ", )
#df_codebook.to_excel("manifesto_codebook1.xlsx")




##### merge text + core data + meta data + codes

## merge df_text with df_codebook variables
df_text_codes = pd.merge(df_text, df_codebook, how="left", left_on="cmp_code", right_on="code")
df_text_codes = df_text_codes.rename(columns={"title": "title_variable"})

## add party meta data:
df_core_cl = df_core[["party", "partyname", "partyabbrev", "parfam", "date", "country", "countryname", "oecdmember", "eumember", "coderyear", "coderid", "testresult", "testeditsim", "manual", "id_perm", "corpusversion", "datasetversion"]]
df_core_cl.date = df_core_cl.date.astype(int)
df_core_cl.party = df_core_cl.party.astype(int)
df_text_codes.date = df_text_codes.date.astype(int)
df_text_codes.party = df_text_codes.party.astype(int)
df_text_meta = pd.merge(df_text_codes, df_core_cl, how="left", on=["party", "date"])

## add more metadata from df_manif_meta_cl (language, title, ...)
df_manif_meta_cl_cl = df_manif_meta_cl[["manifesto_id", "title", "language", "source", "handbook", "election_date", "url_original"]].copy()
df_text_meta = pd.merge(df_text_meta, df_manif_meta_cl_cl, how="left", left_on="key", right_on="manifesto_id")
df_text_meta = df_text_meta.rename(columns={"title_y": "title_manifesto", "title_x": "title_variable"})  # "key": "manifesto_id"

## add iso for country and languages
country_iso_map = {
       'Armenia': "arm", 'Bosnia-Herzegovina': "bih", 'Bulgaria': "bgr", 'Spain': "esp", 'Croatia': "hrv",
       'Czech Republic': "cze", 'Denmark': "dnk", 'Belgium': "bel", 'Netherlands': "nld",
       'South Africa': "zaf", 'United Kingdom': "gbr", 'Ireland': "irl", 'United States': "usa",
       'Canada': "can", 'Australia': "aus", 'New Zealand': "nzl", 'Israel': "isr", 'Poland': "pol",
       'Slovakia': "svk", 'Estonia': "est", 'Finland': "fin", 'Luxembourg': "lux", 'France': "fra",
       'Switzerland': "che", 'Georgia': "geo", 'Italy': "ita", 'Germany': "deu", 'Austria': "aut", 'Greece': "grc",
       'Cyprus': "cyp", 'Hungary': "hun", 'Iceland': "isl", 'Japan': "jpn", 'South Korea': "kor", 'Latvia': "lva",
       'Lithuania': "ltu", 'North Macedonia': "mkd", 'Montenegro': "mne", 'Norway': "nor", 'Portugal': "prt",
       'Moldova': "mda", 'Romania': "rou", 'Russia': "rus", 'Serbia': "srb", 'Slovenia': "svn", 'Mexico': "mex",
       'Sweden': "swe", 'Turkey': "tur", 'Ukraine': "ukr"
}
language_iso2_map = {
       'armenian': "hye", 'bosnian': "bos", 'bos-cyrillic': "bosnian-cyrillic", 'bulgarian': "bul", 'catalan': "cat",
       'croatian': "hrv", 'czech': "ces", 'danish': "dan", 'dutch': "nld", 'english': "eng", 'estonian': "est",
       'finnish': "fin", 'french': "fra", 'galician': "glg", 'georgian': "kat", 'german': "deu", 'greek': "ell",
       'hebrew': "heb", 'hungarian': "hun", 'icelandic': "isl", 'italian': "ita", 'japanese': "jpn",
       'korean': "kor", 'latvian': "lav", 'lithuanian': "lit", 'macedonian': "mkd", 'montenegrin': "cnr",
       'norwegian': "nor", 'polish': "pol", 'portuguese': "por", 'romanian': "ron", 'russian': "rus",
       'serbian-cyrillic': 'srp-cyrillic', 'serbian-latin': 'srp-latin', 'slovak': "slk", 'slovenian': "slv",
       'spanish': "spa", 'swedish': "swe", 'turkish': "tur", 'ukrainian': "ukr"
}
language_iso1_map = {
       'armenian': "hy", 'bosnian': "bs", 'bosnian-cyrillic': "bosnian-cyrillic", 'bulgarian': "bg", 'catalan': "ca",
       'croatian': "hr", 'czech': "cs", 'danish': "da", 'dutch': "nl", 'english': "en", 'estonian': "et",
       'finnish': "fi", 'french': "fr", 'galician': "gl", 'georgian': "ka", 'german': "de", 'greek': "el",
       'hebrew': "he", 'hungarian': "hu", 'icelandic': "is", 'italian': "it", 'japanese': "ja",
       'korean': "ko", 'latvian': "lv", 'lithuanian': "lt", 'macedonian': "mk", 'montenegrin': "cnr",
       'norwegian': "no", 'polish': "pl", 'portuguese': "pt", 'romanian': "ro", 'russian': "ru",
       'serbian-cyrillic': 'sr-cyrillic', 'serbian-latin': 'sr-latin', 'slovak': "sk", 'slovenian': "sl",
       'spanish': "es", 'swedish': "sv", 'turkish': "tr", 'ukrainian': "uk"
}

df_text_meta["country_iso"] = df_text_meta.countryname.map(country_iso_map)
df_text_meta["language_iso"] = df_text_meta.language.map(language_iso1_map)

## add sentence id
df_text_meta["sentence_id"] = range(len(df_text_meta))


## explore final df_text_meta dataset
df_text_meta_cl = df_text_meta[['text', 'language', 'language_iso', 'cmp_code', 'domain_code', 'domain_name', 'code', 'variable_name', 'title_variable', 'label',
                                'type', 'manual', 'description_md',
                                'manifesto_id', 'sentence_id', 'date', 'party', 'partyname', 'partyabbrev', 'parfam',
                                'countryname', 'country_iso', 'country', 'coderyear',
                                'coderid', 'testresult', 'testeditsim'
                                ]]  # 'url_original', 'id_perm', 'oecdmember', 'eumember', # 'eu_code', 'corpusversion', 'datasetversion', 'title_manifesto', 'source', 'key' == 'manifesto_id'
df_text_meta_cl.rename(columns={"countryname": "country_name"}, inplace=True)

## filter only languages I want to analyse
print(df_text_meta_cl.columns)
print(df_text_meta_cl.language.unique())
print(df_text_meta_cl.language.value_counts())


## write to disk
os.getcwd()
df_text_meta_cl.to_csv("data-raw/manifesto_all_2021a.csv", index=True, index_label="idx")









### read disk

# load textual datasets from disk
#df_text_nl = pd.read_csv("manifesto/texts/manif_dutch_texts.csv", index_col="Unnamed: 0")

### citation data
## api_get_core_citation
"""url = "https://manifesto-project.wzb.eu/tools/api_get_core_citation.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
params = dict(api_key=api_key, key=corp_v)
response = requests.get(url=url, params=params)
data_citation_core = response.json()
## api_get_corpus_citation
url = "https://manifesto-project.wzb.eu/tools/api_get_corpus_citation.json"  # ?api_key=f4aca6cabddc5170b7aaf41f8119af45
params = dict(api_key=api_key, key=meta_v)
response = requests.get(url=url, params=params)
data_citation_corpus = response.json()"""


