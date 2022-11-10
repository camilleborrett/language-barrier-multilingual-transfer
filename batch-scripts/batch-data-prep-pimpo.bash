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

# general translation in one language direction
dataset='pimpo_samp'
study_date='221110'
nmt_model='m2m_100_418M'  # m2m_100_418M, m2m_100_1.2B
batch_size=48  # 16, 64
max_length=160
language_target='en'
text_col=''

# translate
python ./data-preparation/translation-general.py --dataset $dataset --language $language_target --nmt_model $nmt_model --batch_size $batch_size --max_length $max_length  &> ./data-preparation/logs-translate-$dataset-$nmt_model-$study_date.txt

# embed
python ./data-preparation/embed.py --dataset $dataset --nmt_model $nmt_model  &> ./data-preparation/logs-embed-$dataset-$nmt_model-$study_date.txt

# tfidf
python ./data-preparation/tfidf-prep.py --dataset $dataset --nmt_model $nmt_model  &> ./data-preparation/logs-tfidf-$dataset-$nmt_model-$study_date.txt




