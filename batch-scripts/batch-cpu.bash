#!/bin/bash
# Set batch job requirements
#SBATCH -t 5:00:00
#SBATCH --partition=thin
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=cpu1
#SBATCH --ntasks=32

# Loading modules for Snellius
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# set correct working directory
cd ./multilingual-repo

# install packages
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
#pip uninstall -y codecarbon
#python -m spacy download en_core_web_md

## scenarios
# "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
# "tfidf", "embeddings-en", "embeddings-multi"


## no-nmt-single
# tfidf
# embeddings-en
# embeddings-multi
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "no-nmt-single" --model "logistic" --vectorizer "embeddings-multi" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026  &> ./results/manifesto-8/logs/hp-logistic-no-nmt-single-embeddings-multi-logs.txt


## one2anchor
# tfidf
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "one2anchor" --model "logistic" --vectorizer "tfidf" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-one2anchor-tfidf-logs.txt
# embeddings-en
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "one2anchor" --model "logistic" --vectorizer "embeddings-en" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-one2anchor-embeddings-en-logs.txt
# embeddings-multi
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "one2anchor" --model "logistic" --vectorizer "embeddings-multi" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-one2anchor-embeddings-multi-logs.txt


## one2many
# tfidf
# embeddings-en
# embeddings-multi
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "one2many" --model "logistic" --vectorizer "embeddings-multi" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-one2many-embeddings-multi-logs.txt


## no-nmt-many
# tfidf
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "no-nmt-many" --model "logistic" --vectorizer "tfidf" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-no-nmt-many-tfidf-logs.txt
# embeddings-en
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "no-nmt-many" --model "logistic" --vectorizer "embeddings-en" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-no-nmt-many-embeddings-en-logs.txt
# embeddings-multi
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "no-nmt-many" --model "logistic" --vectorizer "embeddings-multi" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-no-nmt-many-embeddings-multi-logs.txt


## many2anchor
# tfidf
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "many2anchor" --model "logistic" --vectorizer "tfidf" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-many2anchor-tfidf-logs.txt
# embeddings-en
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "many2anchor" --model "logistic" --vectorizer "embeddings-en" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-many2anchor-embeddings-en-logs.txt
# embeddings-multi
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "many2anchor" --model "logistic" --vectorizer "embeddings-multi" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-many2anchor-embeddings-multi-logs.txt


## many2many
# tfidf
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "many2many" --model "logistic" --vectorizer "tfidf" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-many2many-tfidf-logs.txt
# embeddings-en
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "many2many" --model "logistic" --vectorizer "embeddings-en" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-many2many-embeddings-en-logs.txt
# embeddings-multi
python analysis-classical-hyperparams.py --n_trials 10 --n_trials_sampling 7 --n_trials_pruning 7 --n_cross_val_hyperparam 2 \
                                         --dataset "manifesto-8" --languages "en" "de" "es" "fr" "tr" "ru" "ko" --language_anchor "en" --language_train "en" \
                                         --augmentation_nmt "many2many" --model "logistic" --vectorizer "embeddings-multi" --method "classical_ml" \
                                         --sample_interval 300 --hyperparam_study_date 20221026 &> ./results/manifesto-8/logs/hp-logistic-many2many-embeddings-multi-logs.txt


#python analysis-classical-hyperparams.py --n_trials 60 --n_trials_sampling 30 --n_trials_pruning 40 --n_cross_val_hyperparam 2 --context --dataset "sentiment-news-econ" --sample_interval 100 500 1000 --method "classical_ml" --model "logistic" --hyperparam_study_date 20220713 --vectorizer "tfidf"
#python analysis-classical-run.py --dataset "sentiment-news-econ" --sample_interval 100 500 1000 --method "classical_ml" --model "logistic" --n_cross_val_final 3 --hyperparam_study_date 20220713 --vectorizer "tfidf" --zeroshot
