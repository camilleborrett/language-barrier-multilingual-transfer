
# overall statistics on the two datasets
import pandas as pd
df_manifesto = pd.read_csv("./data-clean/df_manifesto_all.zip")
df_pimpo = pd.read_csv("./data-clean/df_pimpo_all.zip")

lang_manifesto = df_manifesto.language_iso.unique().tolist()
lang_pimpo = df_pimpo.language_iso.unique().tolist()
lang_all = set(lang_manifesto + lang_pimpo)

country_manifesto = df_manifesto.country_iso.unique().tolist()
country_pimpo = df_pimpo.country_iso.unique().tolist()
country_all = set(country_manifesto + country_pimpo)

# second analysis
df_pimpo_nli_en = pd.read_csv(f"./results/pimpo/df_pimpo_pred_immigration_nli_large_long_en_500samp_en_221111.zip")  # _{method}_{model_size}_{hypothesis}_{vectorizer}_{max_sample_lang}samp_{language_str}_{date}.zip")  # usecols=lambda x: x not in ["text_concat", "text_concat_embed_multi", "text_trans_concat", "text_trans_concat_embed_en", "text_trans_concat_tfidf", "selection", "certainty_selection", "topic", "certainty_topic", "direction", "certainty_direction", "rn", "cmp_code"]
df_pimpo_nli_en_de = pd.read_csv(f"./results/pimpo/df_pimpo_pred_immigration_nli_large_long_en_500samp_en_de_221111.zip")  # _{method}_{model_size}_{hypothesis}_{vectorizer}_{max_sample_lang}samp_{language_str}_{date}.zip")  # usecols=lambda x: x not in ["text_concat", "text_concat_embed_multi", "text_trans_concat", "text_trans_concat_embed_en", "text_trans_concat_tfidf", "selection", "certainty_selection", "topic", "certainty_topic", "direction", "certainty_direction", "rn", "cmp_code"]

df_pimpo_nli_en_train = df_pimpo_nli_en[df_pimpo_nli_en.label_text_pred.isna()]
df_pimpo_nli_en_de_train = df_pimpo_nli_en_de[df_pimpo_nli_en_de.label_text_pred.isna()]




