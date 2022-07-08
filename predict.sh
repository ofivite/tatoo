#!/bin/bash

# model
EXPERIMENT_ID=
RUN_ID=""
BATCH_SIZE=4096
CHECKPOINT=""

# dataset
VS_TYPE=
DATASET_TYPE=

DATASET_VS= #  TTToSemiLeptonic    DYJetsToLL_M-50-amcatnloFXFX_ext2    SM_v2
N_FILES_VS=-1 

DATASET_TAU= # GluGluHToTauTau_M125    SM_v2
N_FILES_TAU=-1

python predict.py experiment_id=$EXPERIMENT_ID run_id=$RUN_ID checkpoint=$CHECKPOINT path_to_dataset=datasets batch_size=$BATCH_SIZE n_files=$N_FILES_TAU dataset_name=$DATASET_TAU dataset_type=$DATASET_TYPE tau_type=tau
python predict.py experiment_id=$EXPERIMENT_ID run_id=$RUN_ID checkpoint=$CHECKPOINT path_to_dataset=datasets batch_size=$BATCH_SIZE n_files=$N_FILES_VS dataset_name=$DATASET_VS dataset_type=$DATASET_TYPE tau_type=$VS_TYPE 


