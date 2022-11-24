#!/bin/bash
# Set batch job requirements
#SBATCH -t 20:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=std

# Loading modules for Snellius
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# set correct working directory
cd ./multilingual-repo

# install packages
pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

## for local run
#bash ./batch-scripts/batch-gpu.bash
## scenarios
# "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
# "tfidf", "embeddings-en", "embeddings-multi"

study_date=221111
sample=500
#n_trials=10
#n_trials_sampling=7
#n_trials_pruning=7
#n_cross_val_hyperparam=2
n_cross_val_final=3
model='transformer'   # exact model automatically chosen in script
method='standard_dl'  # standard_dl, nli
dataset='pimpo_samp_a1'  # manifesto-8
nmt_model='m2m_100_418M'  # m2m_100_418M, m2m_100_1.2B
max_epochs=50  # 20, 50
max_length=256



### scenario loops
# these two scenarios only works for embeddings-multi, run faster
translation_lst='no-nmt-single one2many'
vectorizer_lst='embeddings-multi'

for translation in $translation_lst
do
  for vectorizer in $vectorizer_lst
  do
    python analysis-transf-run-a1.py --n_cross_val_final $n_cross_val_final \
           --dataset $dataset --languages 'sv' 'no' 'da' 'fi' 'nl' 'es' 'de' 'en' 'fr' --language_anchor "en" --language_train "en" --nmt_model $nmt_model \
           --augmentation_nmt $translation --model $model --vectorizer $vectorizer --method $method --max_epochs $max_epochs --max_length $max_length \
           --sample_interval $sample --hyperparam_study_date $study_date  &> ./results/$dataset/logs/run-$model-$method-$translation-$vectorizer-$sample-$dataset-$nmt_model-$study_date-logs.txt
    echo Final run done for scenario: $translation $vectorizer
  done
done

## remaining scenarios
translation_lst='one2anchor no-nmt-many many2anchor many2many'
vectorizer_lst='embeddings-en embeddings-multi'

echo Starting training loop
for translation in $translation_lst
do
  for vectorizer in $vectorizer_lst
  do
    python analysis-transf-run-a1.py --n_cross_val_final $n_cross_val_final \
           --dataset $dataset --languages 'sv' 'no' 'da' 'fi' 'nl' 'es' 'de' 'en' 'fr' --language_anchor "en" --language_train "en" --nmt_model $nmt_model \
           --augmentation_nmt $translation --model $model --vectorizer $vectorizer --method $method --max_epochs $max_epochs --max_length $max_length \
           --sample_interval $sample --hyperparam_study_date $study_date  &> ./results/$dataset/logs/run-$model-$method-$translation-$vectorizer-$sample-$dataset-$nmt_model-$study_date-logs.txt
    echo Final run done for scenario: $translation $vectorizer
  done
done


echo Entire script done



## manual runs
:"
python analysis-transf-run-a1.py --n_cross_val_final 3 \
           --dataset 'pimpo_samp_a1' --languages 'sv' 'no' 'da' 'fi' 'nl' 'es' 'de' 'en' 'fr' --language_anchor 'en' --language_train 'en' --nmt_model 'm2m_100_418M' \
           --augmentation_nmt 'many2many' --model 'transformer' --vectorizer 'embeddings-multi' --method 'standard_dl' --max_epochs 50 --max_length 256 \
           --sample_interval 500 --hyperparam_study_date 221111  &> ./results/pimpo_samp_a1/logs/run-transformer-standard_dl-many2many-embeddings-multi-500-pimpo_samp_a1-m2m_100_418M-221111-logs.txt

python analysis-transf-run-a1.py --n_cross_val_final 3 \
           --dataset 'pimpo_samp_a1' --languages 'sv' 'no' 'da' 'fi' 'nl' 'es' 'de' 'en' 'fr' --language_anchor 'en' --language_train 'en' --nmt_model 'm2m_100_418M' \
           --augmentation_nmt 'many2many' --model 'transformer' --vectorizer 'embeddings-multi' --method 'nli' --max_epochs 20 --max_length 256 \
           --sample_interval 500 --hyperparam_study_date 221111  &> ./results/pimpo_samp_a1/logs/run-transformer-nli-many2many-embeddings-multi-500-pimpo_samp_a1-m2m_100_418M-221111-logs.txt
"