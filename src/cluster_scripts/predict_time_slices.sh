#!/bin/bash
#SBATCH --job-name="BNN training"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=3G
#SBATCH --array=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

module load Conda/miniconda/3
eval "$(conda shell.bash hook)"
conda activate /home/tandermann/miniconda3/envs/paleoveg

data_folder=data/time_slice_features
timepoint=${SLURM_ARRAY_TASK_ID}
weight_pickle=$1
accthrestbl_file=$2
noise_p=0.0
noise_t=0.0
noise_e=0.5
burnin=500

python runner_predict_single_time_slice.py $data_folder $weight_pickle $timepoint $accthrestbl_file $noise_p $noise_t $noise_e $burnin


