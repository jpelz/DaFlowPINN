#!/bin/bash

#SBATCH -o /scratch/jpelz/slurm_outs/halfcylinder.%j.%N.out   # Output-File
#SBATCH -D /scratch/jpelz/ma-pinns/final_sims/                 # Working Directory
#SBATCH -J PINN_Halfcylinder	# Job Name
#SBATCH --ntasks=1 		# Anzahl Prozesse P (CPU-Cores) 
#SBATCH --cpus-per-task=5	# Anzahl CPU-Cores pro Prozess P
#SBATCH --gres=gpu:1 # Anzahl GPUs pro Prozess P
#SBATCH --mem=200G              # GiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=05:00:00 # Erwartete Laufzeit

#Auf GPU-Knoten rechnen:
#SBATCH --partition=gpu,gpu_short,scioi_gpu,scioi_a100nv

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=###

source /scratch/jpelz/envs/p38gpu/bin/activate

python /scratch/jpelz/ma-pinns/main.py $1
#python post2.py
