#!/bin/bash
#Set batch job requirements
#SBATCH -t 10:00:00
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

# trainslate train and test
python ./data-preparation/translation.py &> ./data-preparation/translate-logs_221101.txt

# embed
python ./data-preparation/embed.py &> ./data-preparation/embed-logs_221101.txt

# tfidf
python ./data-preparation/tfidf-prep.py &> ./data-preparation/tfidf-logs_221101.txt




