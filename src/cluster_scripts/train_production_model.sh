#!/bin/bash
#SBATCH --job-name="BNN training"
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=3G

module load Conda/miniconda/3
eval "$(conda shell.bash hook)"
conda activate /home/tandermann/miniconda3/envs/paleoveg

Ncurrent=1655
Npaleo=331
nnodes=32_8
biotic_features=1
abiotic_features=1
continue_run=1
cv=0
final_model=1
sum_pooling=0
max_pooling=0
pklpath=results/n_current_1655_n_paleo_331_nnodes_32_8_biotic_1_abiotic_1_sumpool_0_maxpool_0/continued_cv1_current_1655_paleo_331_p1_h0_l32_8_s1_binf_1234.pkl
outdir=results/production_model_thirdbest_model

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

