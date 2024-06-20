#!/bin/sh

#SBATCH --job-name="mordm_db"
#SBATCH --partition=compute
#SBATCH --account=Education-TPM-MSc-EPA
#SBATCH --time=03:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=output_file.out
#SBATCH --error=error_file.err

# Load required modules
module load 2023r1
module load openmpi
module load python
module load py-pip

# Uninstall old numpy version
pip uninstall -y numpy

# Install required packages
pip install --user --upgrade numpy==1.26.4 ema_workbench networkx

# Export user base path for pip packages
export PYTHONUSERBASE=$HOME/.local
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# Check installation
python3 -c "import networkx; print(networkx.__version__)"

# Run the script using mpiexec
mpiexec -n 10 python3 MORDM_test_db.py
