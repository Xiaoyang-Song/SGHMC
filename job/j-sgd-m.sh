#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=SGD-M
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/SGHMC/out/j-sgd-m.log

python src/main_sgd.py --dset='mnist' --opt='sgd-m' --lr=1e-3 > /scratch/sunwbgt_root/sunwbgt98/xysong/SGHMC/out/sgd-m-m.log
python src/main_sgd.py --dset='fashion-mnist' --opt='sgd-m' --lr=1e-3 > /scratch/sunwbgt_root/sunwbgt98/xysong/SGHMC/out/sgd-m-fm.log