

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


import pandas as pd
import numpy as np
import spacy
import tqdm

SEED_GLOBAL = 42

## load dataset to process
if "manifesto-8" in DATASET:
  #df_cl = pd.read_csv("./data-clean/df_manifesto_all.csv", index_col="idx")
  df_train = pd.read_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}_embed.csv", index_col="idx")
  df_test = pd.read_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}_embed.csv", index_col="idx")
else:
  raise Exception(f"Dataset name not found: {DATASET}")

# for some reason, some translations are NaN
# hopefully corrected when final run with bigger translation model
df_train = df_train[~df_train.text_original_trans.isna()]
df_test = df_test[~df_test.text_original_trans.isna()]


## functions for lemmatization and stopword removal
def lemmatize_and_stopwords(text_lst):
    texts_lemma = []
    for doc in nlp.pipe(text_lst, n_process=1):  # disable=["tok2vec", "ner"] "tagger", "attribute_ruler", "parser",
        #doc_lemmas = " ".join([token.lemma_ for token in doc if not token.is_stop])  #token.lemma_ not in nlp.Defaults.stop_words
        # if else in case all tokens are deleted due to stop word removal
        doc_lemmas = [token.lemma_ for token in doc if not token.is_stop]
        if not any(pd.isna(doc_lemmas)):
            doc_lemmas = " ".join(doc_lemmas)
            texts_lemma.append(doc_lemmas)
        else:
            print(doc)
            texts_lemma.append(doc.text)
    return texts_lemma

def remove_stopwords(text_lst, stopword_lst=None):
    texts_lst_cl = []
    for text in text_lst:
        text_cl = ' '.join([word for word in text.split() if word not in stopword_lst])
        texts_lst_cl.append(text_cl)
    return texts_lst_cl

# Spacy models available for differen languages https://spacy.io/usage/models
spacy_models_dic = {"ko": "ko_core_news_md", "en": "en_core_web_md", "ru": "ru_core_news_md",
                    "es": "es_core_news_md", "de": "de_core_news_md", "fr": "fr_core_news_md",
                    "tr": "turkish-stopwords"  # https://github.com/explosion/spaCy/tree/master/spacy/lang/tr
}

## prepare test set
df_test_prep_lst = []
for group_name, group_df in tqdm.tqdm(df_test.groupby(by="language_iso_trans", as_index=False, group_keys=False)):
    np.random.seed(SEED_GLOBAL)
    print(group_name)
    #group_df = group_df.sample(n=5, random_state=42)  # for testing
    if group_name != "tr":
        nlp = spacy.load(spacy_models_dic[group_name])
        group_df["text_original_trans_tfidf"] = lemmatize_and_stopwords(group_df.text_original_trans)
        df_test_prep_lst.append(group_df)
    elif group_name == "tr":
        stopwords_tr = pd.read_json("https://raw.githubusercontent.com/stopwords-iso/stopwords-tr/master/stopwords-tr.json")
        group_df["text_original_trans_tfidf"] = remove_stopwords(group_df.text_original_trans, stopword_lst=stopwords_tr)
        df_test_prep_lst.append(group_df)
df_test_prep = pd.concat(df_test_prep_lst)

## prepare train set
df_train_prep_lst = []
for group_name, group_df in tqdm.tqdm(df_train.groupby(by="language_iso_trans", as_index=False, group_keys=False)):
    np.random.seed(SEED_GLOBAL)
    print(group_name)
    #group_df = group_df.sample(n=5, random_state=42)  # for testing
    if group_name != "tr":
        nlp = spacy.load(spacy_models_dic[group_name])
        group_df["text_original_trans_tfidf"] = lemmatize_and_stopwords(group_df.text_original_trans)
        df_train_prep_lst.append(group_df)
    elif group_name == "tr":
        stopwords_tr = pd.read_json("https://raw.githubusercontent.com/stopwords-iso/stopwords-tr/master/stopwords-tr.json")
        group_df["text_original_trans_tfidf"] = remove_stopwords(group_df.text_original_trans, stopword_lst=stopwords_tr)
        df_train_prep_lst.append(group_df)
df_train_prep = pd.concat(df_train_prep_lst)

# test output visually
test = df_train_prep[["language_iso_trans", "text_original_trans", "text_original_trans_tfidf"]]

# remove duplicates again here ? No, if tfidf preparation introduces duplicates, its an issue of tfidf reduction
"""test1 = df_train_prep[df_train_prep.text_original_trans_tfidf.duplicated(keep=False)]
test2 = df_test_prep[df_test_prep.text_original_trans_tfidf.duplicated(keep=False)]
# also remove random duplicate overlap between train and test
df_concat = pd.concat([df_train_prep, df_test_prep])
df_train_prep
sum(df_train_prep.duplicated(subset=["text_original_trans_tfidf"]))
sum(df_test_prep.duplicated(subset=["text_original_trans_tfidf"]))
sum(df_concat.duplicated(subset=["text_original_trans_tfidf"]))"""

# test if introduced empty strings / NAs
# some empty strings introduced through bad translations
print(len(df_train_prep))
print(len(df_test_prep))
df_train_prep = df_train_prep[~df_train_prep.text_original_trans_tfidf.isna()]
df_test_prep = df_test_prep[~df_test_prep.text_original_trans_tfidf.isna()]
df_train_prep = df_train_prep[df_train_prep.text_original_trans_tfidf != ""]
df_test_prep = df_test_prep[df_test_prep.text_original_trans_tfidf != ""]
print(len(df_train_prep))
print(len(df_test_prep))


#### write to disk
df_train_prep.to_csv(f"./data-clean/df_{DATASET}_train_trans_{NMT_MODEL}_embed_tfidf.csv", index=False)
df_test_prep.to_csv(f"./data-clean/df_{DATASET}_test_trans_{NMT_MODEL}_embed_tfidf.csv", index=False)


print("Script done.")





## test default sklearn tokenizer for Korean
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
"""import re
import spacy

# standard tokenizer from sklearn
token_pattern = r"(?u)\b\w\w+\b"
token_pattern = re.compile(token_pattern)

# tokenizer with standard re on full text
text = "스마트팜 혁신밸리 내 청년창업 교육과정을 통해 스마트팜 전문인력을 양성하고, 스마트농업 특수대학원을 설립"
print(token_pattern.findall(text))

# tokenization on lemmatized text
nlp = spacy.load("ko_core_news_md")
text_lemmatized = " ".join([str(token.lemma_) for token in nlp(text) if not token.is_stop])

# lemmatization creates smaller semantic units, which is good. the "+" from spacy lemmatization gets deleted by sk tokenizer, but seems to preserve semantically split units
# Korean seems to have composite words like German
print(text_lemmatized)
print(token_pattern.findall(text_lemmatized))

"""