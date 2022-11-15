#!/bin/bash
# Set batch job requirements
#SBATCH -t 30:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=pimpo

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
method_lst='nli standard_dl'  # nli, standard_dl, (transformer-embed)
vectorizer_lst='en multi'  # en, multi
model_size_lst='base large'
languages_lst="en en-de en-de-sv-fr"  # "en", "en-de", "en-de-sv-fr"
task_lst='immigration'  # immigration, integration
max_sample_lang=500
max_epochs=50  # only taken into account for standard_dl
max_length=320
study_date=221111
hypothesis='long'
nmt_model='m2m_100_418M'  # m2m_100_418M, m2m_100_1.2B


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
          python analysis-pimpo-run.py --languages $lang --method $method --hypothesis $hypothesis --vectorizer $vectorizer --task $task --model_size $model_size \
                                   --max_sample_lang $max_sample_lang --study_date $study_date --nmt_model $nmt_model --max_epochs $max_epochs \
                                   --max_length $max_length  &> ./results/pimpo/logs/logs-$method-$model_size-$hypothesis-$vectorizer-$max_sample_lang-$task-$lang-$study_date.txt
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
