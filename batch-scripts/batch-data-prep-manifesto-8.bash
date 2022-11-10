#!/bin/bash
#Set batch job requirements
#SBATCH -t 1:00:00
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

dataset='manifesto-8_samp'
dataset_train='manifesto-8_samp_train'
dataset_test='manifesto-8_samp_test'
nmt_model='m2m_100_418M'  # m2m_100_418M, m2m_100_1.2B
batch_size=48  # 16, 64
max_length=160
study_date='221110'


# trainslate train and test
python ./data-preparation/translation-manifesto-8.py --dataset $dataset --nmt_model $nmt_model --batch_size $batch_size --max_length $max_length  &> ./data-preparation/logs-translate-$dataset-$nmt_model-$study_date.txt

# embed
python ./data-preparation/embed.py --dataset $dataset_train --nmt_model $nmt_model  &> ./data-preparation/logs-embed-$dataset_train-$nmt_model-$study_date.txt
python ./data-preparation/embed.py --dataset $dataset_test --nmt_model $nmt_model  &> ./data-preparation/logs-embed-$dataset_test-$nmt_model-$study_date.txt

# tfidf
python ./data-preparation/tfidf-prep.py --dataset $dataset_train --nmt_model $nmt_model  &> ./data-preparation/logs-tfidf-$dataset_train-$nmt_model-$study_date.txt
python ./data-preparation/tfidf-prep.py --dataset $dataset_test --nmt_model $nmt_model  &> ./data-preparation/logs-tfidf-$dataset_test-$nmt_model-$study_date.txt




