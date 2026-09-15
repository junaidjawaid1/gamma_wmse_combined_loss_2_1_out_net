#!/bin/bash
#SBATCH -s
#SBATCH -n 1
#SBATCH -o /.../out/gamma_eval_1mm_1prc.out
#SBATCH -J gamma_eval_1mm_1prc
#SBATCH -p cuda
#SBATCH -c 16
#SBATCH --gres=gpu:fast

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

GRB_VER=1001


srun python "$@"
