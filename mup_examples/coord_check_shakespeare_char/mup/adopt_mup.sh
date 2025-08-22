#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=48:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A datascience_collab

module use /soft/modulefiles
module load conda
conda activate

cd /eagle/projects/datascience_collab/agupta
source venv_nanogpt2/bin/activate

cd nanoGPT-HPTransfer
bash mup_examples/coord_check_shakespeare_char/mup/run_adopt_mup.sh
