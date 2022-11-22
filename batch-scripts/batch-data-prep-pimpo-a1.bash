#!/bin/bash
#Set batch job requirements
#SBATCH -t 25:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=prep

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./multilingual-repo

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

# train test split done locally, easier to inspect, less to upload, no GPU necessary
#python ./data-preparation/data-prep-manifesto.py

dataset='pimpo_samp_a1'
dataset_train='pimpo_samp_a1_train'
dataset_test='pimpo_samp_a1_test'
nmt_model='m2m_100_418M'  # m2m_100_418M, m2m_100_1.2B
batch_size=48  # 16, 48
max_length=200
study_date='221111'


# trainslate train and test
python ./data-preparation/translation-analysis-1.py --dataset $dataset --nmt_model $nmt_model --batch_size $batch_size --max_length $max_length  &> ./data-preparation/logs/logs-translate-$dataset-$nmt_model-$study_date.txt

# embed
python ./data-preparation/embed.py --dataset $dataset_train --nmt_model $nmt_model  &> ./data-preparation/logs/logs-embed-$dataset_train-$nmt_model-$study_date.txt
python ./data-preparation/embed.py --dataset $dataset_test --nmt_model $nmt_model  &> ./data-preparation/logs/logs-embed-$dataset_test-$nmt_model-$study_date.txt

# tfidf
python ./data-preparation/tfidf-prep.py --dataset $dataset_train --nmt_model $nmt_model  &> ./data-preparation/logs/logs-tfidf-$dataset_train-$nmt_model-$study_date.txt
python ./data-preparation/tfidf-prep.py --dataset $dataset_test --nmt_model $nmt_model  &> ./data-preparation/logs/logs-tfidf-$dataset_test-$nmt_model-$study_date.txt


# manual run
#python ./data-preparation/tfidf-prep.py --dataset 'pimpo_samp_a1_train' --nmt_model 'm2m_100_418M'  &> ./data-preparation/logs/logs-tfidf-pimpo_samp_a1_train-m2m_100_418M-221111.txt
#python ./data-preparation/tfidf-prep.py --dataset 'pimpo_samp_a1_test' --nmt_model 'm2m_100_418M'  &> ./data-preparation/logs/logs-tfidf-pimpo_samp_a1_test-m2m_100_418M-221111.txt


