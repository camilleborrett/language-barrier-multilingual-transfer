#!/bin/bash
# Set batch job requirements
#SBATCH -t 25:00:00
#SBATCH --partition=thin
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=cpu2
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


## generalised loop
method_lst='dl_embed'  # nli, standard_dl, dl_embed
vectorizer_lst='en multi'  # en, multi
model_size_lst='classical'
languages_lst="en en-de en-de-sv-fr"  # "en", "en-de", "en-de-sv-fr"
task_lst='immigration'  # immigration, integration
max_sample_lang=500
#max_epochs=50  # only taken into account for standard_dl
#max_length=320  # only for transformers
study_date=221111
hypothesis='long'
nmt_model='m2m_100_1.2B'  # m2m_100_418M, m2m_100_1.2B
# hp-search
n_trials=30  #30
n_trials_sampling=15  #15
n_trials_pruning=15  #15
n_cross_val_hyperparam=2


for model_size in $model_size_lst
do
  for task in $task_lst
  do
    for method in $method_lst
    do
      for vectorizer in $vectorizer_lst
      do
        for lang in $languages_lst
        do
          python analysis-classical-hyperparams-a2.py --n_trials $n_trials --n_trials_sampling $n_trials_sampling --n_trials_pruning $n_trials_pruning --n_cross_val_hyperparam $n_cross_val_hyperparam \
                   --languages $lang --method $method --vectorizer $vectorizer --task $task \
                   --max_sample_lang $max_sample_lang --study_date $study_date \
                   --nmt_model $nmt_model  &> ./results/pimpo/logs/logs-hp-$method-$model_size-$hypothesis-$vectorizer-$max_sample_lang-$task-$lang-$study_date.txt
          python analysis-classical-run-a2.py --languages $lang --method $method --hypothesis $hypothesis --vectorizer $vectorizer --task $task --model_size $model_size \
                   --max_sample_lang $max_sample_lang --study_date $study_date \
                   --nmt_model $nmt_model  &> ./results/pimpo/logs/logs-run-$method-$model_size-$hypothesis-$vectorizer-$max_sample_lang-$task-$lang-$study_date.txt
          echo Run done for scenario: $method $vectorizer $task $lang
        done
      done
    done
  done
done

echo Entire script done


# for local run
#bash ./batch-scripts/batch-gpu.bash
# for manual commandline tests
#python analysis-pimpo-run.py --languages "en" --method "nli" --hypothesis "long" --vectorizer "en" --task 'immigration' --max_sample_lang 10000 --study_date 221111 --mt_model 'm2m_100_418M' --max_epochs 20 --max_length 256  &> ./results/pimpo/logs/logs-nli-long-en-500-immigration-en-221111.txt


