#!/bin/bash
#SBATCH --job-name="BNN training"
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=3G

module load Conda/miniconda/3
eval "$(conda shell.bash hook)"
conda activate /home/tandermann/miniconda3/envs/paleoveg

targetdir=$1

python get_test_acc_across_cv_folds.py $targetdir
