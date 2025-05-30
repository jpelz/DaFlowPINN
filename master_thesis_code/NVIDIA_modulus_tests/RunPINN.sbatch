#!/bin/bash

#SBATCH -o /scratch/jpelz/modulus/myjob.%j.%N.out   # Output-File
#SBATCH -D /scratch/jpelz/modulus/                  # Working Directory
#SBATCH -J Modulus_PTV   	# Job Name
#SBATCH --ntasks=1 		# Anzahl Prozesse P (CPU-Cores) 
#SBATCH --cpus-per-task=1	# Anzahl CPU-Cores pro Prozess P
#SBATCH --gres=gpu:tesla:1	# GPUs anfordern
#SBATCH --mem=3G                # GiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=00:05:00 # Erwartete Laufzeit

#Auf GPU-Knoten rechnen:
#SBATCH --partition=gpu_short

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=#

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="localhost"

source /scratch/jpelz/envs/modulus/bin/activate

#python Test.py

python -u modulusPINN.py
