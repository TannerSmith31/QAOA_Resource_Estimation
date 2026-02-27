#!/bin/bash

#SBATCH --partition=amilan
#SBATCH --job-name=QAOAexperiment
#SBATCH --output=QAOAexperiment.%j.out
#SBATCH --time=10:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
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

NODE_COUNT=$1
WEIGHT_ID=$2

#ensure both arguments are provided
if [[ -z "$NODE_COUNT" || -z "$WEIGHT_ID" ]]; then
  echo "Error: Missing required arguments."
  echo "Usage: sbatch qaoaExperiment_script1.sh <NODE_COUNT> <WEIGHT_ID>"
  exit 1
fi

# Begin Experiment
echo "== START of Job N$NODE_COUNT W$WEIGHT_ID =="


python /projects/$USER/CSCI7000_finalProj/qaoaExperiment.py \
    --numNodes $NODE_COUNT \
    --weightID $WEIGHT_ID \
    --layersToRun 1 2 3 4 5 7 9\
    --distancesToRun 0 -1 \
    --outputDir "/projects/$USER/CSCI7000_finalProj/results/"

echo "== END of Job N$NODE_COUNT W$WEIGHT_ID =="

# Copy everything from the scratch directory to your home directory
# cp -r $SLURM_SCRATCH/* /projects/$USER/CSCI7000_finalProj/results/  #NOT DOING THIS ANYMORE SINCE OUTPUT DIRECTLY TO RESULTS