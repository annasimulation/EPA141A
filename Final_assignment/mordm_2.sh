#!/bin/bash

#SBATCH --job-name="DikeNetworkOptimization"
#SBATCH --time=01:00:00
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=Education-TPM-MSc-EPA
#SBATCH --output=output_18_06.out
#SBATCH --error=error_18_06.err

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip

pip install --user --upgrade ema_workbench

mpiexec -n 24 python3 ema_mpi_model.py
