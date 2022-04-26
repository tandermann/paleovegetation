#!/bin/bash
#SBATCH --job-name="BNN training"
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --array=1,2,3,4,5

module load Conda/miniconda/3
eval "$(conda shell.bash hook)"
conda activate /home/tandermann/miniconda3/envs/paleoveg

Ncurrent=662
Npaleo=331
nnodes=32_8
biotic_features=1
abiotic_features=1
continue_run=1
cv=${SLURM_ARRAY_TASK_ID}
final_model=0
sum_pooling=0
max_pooling=0
pklpath=$1
outdir=results/n_current_${Ncurrent}_n_paleo_${Npaleo}_nnodes_${nnodes}_biotic_${biotic_features}_abiotic_${abiotic_features}_sumpool_${sum_pooling}_maxpool_${max_pooling}

python \
runner_single_chain_recycle_weights.py \
$pklpath \
$nnodes \
$Ncurrent \
$Npaleo \
$biotic_features \
$abiotic_features \
$continue_run \
$cv \
$final_model \
$outdir \
$sum_pooling \
$max_pooling

