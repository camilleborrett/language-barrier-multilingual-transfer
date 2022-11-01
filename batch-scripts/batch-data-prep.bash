#!/bin/bash
#Set batch job requirements
#SBATCH -t 4:00:00
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

nmt_model='m2m_100_1.2B'  # m2m_100_418M, m2m_100_1.2B
batch_size=16  # 16, 64
study_date='221101'
dataset='manifesto-8'

# trainslate train and test
#python ./data-preparation/translation.py --dataset $dataset --nmt_model $nmt_model --batch_size $batch_size  &> ./data-preparation/translate-logs-$dataset-$nmt_model-$study_date.txt

# embed
python ./data-preparation/embed.py --dataset $dataset --nmt_model $nmt_model  &> ./data-preparation/embed-logs-$dataset-$nmt_model-$study_date.txt

# tfidf
python ./data-preparation/tfidf-prep.py --dataset $dataset --nmt_model $nmt_model  &> ./data-preparation/tfidf-logs-$dataset-$nmt_model-$study_date.txt




