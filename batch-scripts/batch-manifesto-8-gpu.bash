#!/bin/bash
#Set batch job requirements
#SBATCH -t 3:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=manifesto-8

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#set correct working directory
cd ./multilingual-repo

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip uninstall -y codecarbon


python data-preparation/translation.py





### old example code from nli paper
## base-nli
#python analysis-transf-hyperparams.py --learning_rate 1e-7 9e-4 --epochs 3 36 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "cap-sotu" --sample_interval 5000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --carbon_tracking
#python analysis-transf-run.py --dataset "cap-sotu" --sample_interval 5000 --method "nli" --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" --n_cross_val_final 3 --hyperparam_study_date 20220502 --carbon_tracking
## base-standard
#python analysis-transf-hyperparams.py --learning_rate 5e-7 9e-4 --epochs 30 100 --batch_size 16 32 --n_trials 14 --n_trials_sampling 7 --n_trials_pruning 5 --n_cross_val_hyperparam 2 --context --dataset "cap-sotu" --sample_interval 5000 --method "standard_dl" --model "microsoft/deberta-v3-base" --carbon_tracking
#python analysis-transf-run.py --dataset "cap-sotu" --sample_interval 5000 --method "standard_dl" --model "microsoft/deberta-v3-base" --n_cross_val_final 3 --hyperparam_study_date 20220503 --carbon_tracking
