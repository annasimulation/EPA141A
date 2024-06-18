#!/bin/bash
#SBATCH --job-name=mordm_job
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --partition=compute

# Load necessary modules
module load .2024rc1
module load python/3.11.6

# Create a virtual environment
python -m venv env
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required Python packages
pip install ema_workbench pandas openpyxl xlrd tqdm  scipy networkx

# Run the Python script
python MORDM_test_db.py
