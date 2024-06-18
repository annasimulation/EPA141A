#!/bin/bash

#SBATCH --job-name="mordm"
#SBATCH --time=01:00:00
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=Education-TPM-MSc-EPA

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip

pip install --user --upgrade ema_workbench

mpiexec -n 48 python3 MORDM_single_run_delftblue.py
