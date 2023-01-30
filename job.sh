#!/usr/bin/env bash

#SBATCH --job-name="train_different_sizes"
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --array=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12        # 1 cpu per node out of 4
#SBATCH --partition=m100_usr_prod
#SBATCH --output=/m100/home/userexternal/ecosta01/dft_for_ising/output/dataset/output_JOB_%j.out
#SBATCH --error=/m100/home/userexternal/ecosta01/dft_for_ising/output/dataset/error_JOB_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=emanuele.costa@unicam.it
#SBATCH --account=IscrC_SMORAGEN

#=============================
# environment

source activate dft_env

echo "Running on "`hostname`

#=============================
# user definitions



#=============================
# running

srun python dft_dataset.py --file_name=train_without_augmentation/unet_pbc --l=16 --pbc --check_2nn --h_max=5.6 --seed=56 --n_dataset=150000 
