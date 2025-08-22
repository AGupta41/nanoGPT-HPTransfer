#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A datascience_collab

module use /soft/modulefiles
module load conda                                                                                                                                           conda activate                                                                                                                                                                                                                                                                                                          cd /eagle/projects/datascience_collab/agupta/2025-07-23-172025/Megatron-DeepSpeed                                                                           source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env && python3 -m pip install deepspeed "git+https://github.com/saforem2/ezpz" --require-virtualenv                                                                                                                                                                                                                                                                                                                       for lr in 0.0005                                                                                                                                            do                                                                                                                                                                  for width in 256 512 1024 2048                                                                                                                              do                                                                                                                                                                  DATA_FILE_LIST=ALCF/data-lists/polaris/dolma.txt TRAIN_TOKENS=7000000000 LR=$lr GRAD_ACC_STEPS=16 HIDDEN=$width NLAYERS=24 MICRO_BATCH=1 OPT=sophiag bash train_aGPT_7B.sh --seed 1234
		cd checkpoints/
                rm -r *
                cd ../.
        done
done
