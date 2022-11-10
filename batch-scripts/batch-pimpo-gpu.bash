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
# for manual commandline tests
python analysis-pimpo.py --languages "en" --method "nli" --hypothesis "long" --vectorizer "en" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-nli-en-long-immigration-en.txt
python analysis-pimpo.py --languages "en" "de" --method "nli" --hypothesis "long" --vectorizer "en" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-nli-en-long-immigration-en-de.txt
python analysis-pimpo.py --languages "en" "de" "sv" "fr" --method "nli" --hypothesis "long" --vectorizer "en" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-nli-en-long-immigration-en-de-sv-fr.txt

python analysis-pimpo.py --languages "en" --method "nli" --hypothesis "long" --vectorizer "multi" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-nli-multi-long-immigration-en.txt
python analysis-pimpo.py --languages "en" "de" --method "nli" --hypothesis "long" --vectorizer "multi" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-nli-multi-long-immigration-en-de.txt
python analysis-pimpo.py --languages "en" "de" "sv" "fr" --method "nli" --hypothesis "long" --vectorizer "multi" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-nli-multi-long-immigration-en-de-sv-fr.txt

python analysis-pimpo.py --languages "en" --method "standard_dl" --hypothesis "long" --vectorizer "en" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-standard_dl-en-long-immigration-en.txt
python analysis-pimpo.py --languages "en" "de" --method "standard_dl" --hypothesis "long" --vectorizer "en" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-standard_dl-en-long-immigration-en-de.txt
python analysis-pimpo.py --languages "en" "de" "sv" "fr" --method "standard_dl" --hypothesis "long" --vectorizer "en" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-standard_dl-en-long-immigration-en-de-sv-fr.txt

python analysis-pimpo.py --languages "en" --method "standard_dl" --hypothesis "long" --vectorizer "multi" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-standard_dl-multi-long-immigration-en.txt
python analysis-pimpo.py --languages "en" "de" --method "standard_dl" --hypothesis "long" --vectorizer "multi" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-standard_dl-multi-long-immigration-en-de.txt
python analysis-pimpo.py --languages "en" "de" "sv" "fr" --method "standard_dl" --hypothesis "long" --vectorizer "multi" --task 'immigration' --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-standard_dl-multi-long-immigration-en-de-sv-fr.txt


task='immigration'
method='nli'
python analysis-pimpo.py --languages "en" --method $method --task $task --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-$task-$method-en.txt
python analysis-pimpo.py --languages "en" "de" --method $method --task $task --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-$task-$method-en-de.txt
python analysis-pimpo.py --languages "en" "de" "nl" --method $method --task $task --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-$task-$method-en-de-nl.txt
python analysis-pimpo.py --languages "en" "de" "sv" "fr" --method $method --task $task --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-$task-$method-en-de-sv-fr.txt

task='integration'
method='nli'
python analysis-pimpo.py --languages "en" --method $method --task $task --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-$task-$method-en.txt
python analysis-pimpo.py --languages "en" "de" --method $method --task $task --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-$task-$method-en-de.txt
python analysis-pimpo.py --languages "en" "de" "nl" --method $method --task $task --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-$task-$method-en-de-nl.txt
python analysis-pimpo.py --languages "en" "de" "sv" "fr" --method $method --task $task --sample 500 --max_epochs 50 --study_date 221103  &> ./results/pimpo/logs/pimpo-logs-$task-$method-en-de-sv-fr.txt


echo Entire script done

