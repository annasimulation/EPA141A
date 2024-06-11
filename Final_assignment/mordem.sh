#!/bin/sh

#SBATCH --job-name="mordm_db"
#SBATCH --partition=compute
#SBATCH --account=research-education-TPM-MSc-EPA
#SBATCH --time=03:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-mpi4py
module load py-scipy
module load py-pip

pip install --user --upgrade ema_workbench

mpiexec -n 1 python3 Ass8.1(MORDM)(5runs).py