#!/bin/bash

#SBATCH --partition=amilan
#SBATCH --job-name=QAOAexperiment
#SBATCH --output=QAOAexperiment.%j.out
#SBATCH --time=07:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tasm1595@colorado.edu

# Enable multithreading for Qiskit Aer + BLAS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export QISKIT_IN_PARALLEL=true

module purge
module load uv

#Activate Qiskit env
source $UV_ENVS/qiskit-env/bin/activate

# Make a results folder in your home if it doesn't exist
mkdir -p /projects/$USER/CSCI7000_finalProj/results/

FILENAME="$1"

#ensure the argument is provided
if [[ -z "$FILENAME" ]]; then
  echo "Error: Missing required argument."
  echo "Usage: sbatch qaoaExperiment_Script2.sh <FILENAME>"
  exit 1
fi

# Begin Experiment
echo "== START of Job FILENAME $FILENAME =="


python /projects/$USER/CSCI7000_finalProj/qaoaExperiment.py \
    --inputFile "$FILENAME" \
    --layersToRun 1 2 3 4 5 7 9\
    --distancesToRun 11 \
    --outputDir "/projects/$USER/CSCI7000_finalProj/results/"

echo "== END of Job FILENAME $FILENAME =="

# Copy everything from the scratch directory to your home directory
# cp -r $SLURM_SCRATCH/* /projects/$USER/CSCI7000_finalProj/results/