#!/bin/bash

#SBATCH -o /scratch/jpelz/modulus/myjob.%j.%N.out   # Output-File
#SBATCH -D /scratch/jpelz/modulus/                 # Working Directory
#SBATCH -J BuildModulus	# Job Name
#SBATCH --nodes=1               # Anzahl Knoten N
#SBATCH --ntasks-per-node=1   # Prozesse n pro Knoten
#SBATCH --mem=1G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=05:00:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=standard

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=###

source /scratch/jpelz/envs/modulus/bin/activate

pip install nvidia-modulus
pip install "pint==0.19.2"
pip install nvidia-modulus.sym --no-build-isolation
