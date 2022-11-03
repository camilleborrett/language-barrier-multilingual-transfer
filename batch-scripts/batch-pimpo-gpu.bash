#!/bin/bash
# Set batch job requirements
#SBATCH -t 3:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=gpu1
#SBATCH --ntasks=32

# Loading modules for Snellius
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# set correct working directory
cd ./multilingual-repo

# install packages
pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# for local run
#bash ./batch-scripts/batch-gpu.bash

python analysis-pimpo.py --languages "en" --task "immigration" --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-immi-en.txt
python analysis-pimpo.py --languages "en" "de" --task "immigration" --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-immi-en-de.txt
python analysis-pimpo.py --languages "en" "de" "sv" "fr" --task "immigration" --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-immi-en-de-sv-fr.txt

python analysis-pimpo.py --languages "en" --task "integration" --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-integ-en.txt
python analysis-pimpo.py --languages "en" "de" --task "integration" --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-integ-en-de.txt
python analysis-pimpo.py --languages "en" "de" "sv" "fr" --task "integration" --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-integ-en-de-sv-fr.txt


echo Entire script done

