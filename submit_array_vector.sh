#!/bin/bash
#SBATCH --chdir=/home/sedm7756/osqp_benchmarks
#SBATCH --job-name=multi-cpu         # job name
#SBATCH --output=slurm-%A_%a.out       # %A = master jobid, %a = array index
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=48:00:00                # walltime (adjust as needed)
#SBATCH --partition=medium            # change to your CPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G                       # 48 GB memory per array task
#SBATCH --array=1-10                    

# ---- environment setup ----
# Make sure conda is available in batch jobs
source /home/sedm7756/miniconda3/etc/profile.d/conda.sh
conda activate rlqp

# avoid oversubscription of OpenMP/MKL threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- list your commands here (index order must match --array range) ----
CMDS=(
  "python learned_osqp/train.py --precision low --types random_qp --sizes 200 --normalize_features"
  "python learned_osqp/train.py --precision low --types random_qp --sizes 200 --adaptive_rho false --normalize_features"

  "python learned_osqp/train.py --precision low --types svm --sizes 15 --normalize_features"
  "python learned_osqp/train.py --precision low --types svm --sizes 15 --adaptive_rho false --normalize_features"

  "python learned_osqp/train.py --precision low --types control --sizes 80 --normalize_features"
  "python learned_osqp/train.py --precision low --types control --sizes 80 --adaptive_rho false --normalize_features"

  "python learned_osqp/train.py --precision low --types lasso --sizes 15 --normalize_features"
  "python learned_osqp/train.py --precision low --types lasso --sizes 15 --adaptive_rho false --normalize_features"

  "python learned_osqp/train.py --precision low --types portfolio --sizes 15 --normalize_features"
  "python learned_osqp/train.py --precision low --types portfolio --sizes 15 --adaptive_rho false --normalize_features"

  # "python learned_osqp/train.py --precision high --types random_qp --sizes 200 --normalize_features"
  # "python learned_osqp/train.py --precision high --types random_qp --sizes 200 --adaptive_rho false --normalize_features"

  # "python learned_osqp/train.py --precision high --types svm --sizes 15 --normalize_features"
  # "python learned_osqp/train.py --precision high --types svm --sizes 15 --adaptive_rho false --normalize_features"

  # "python learned_osqp/train.py --precision high --types control --sizes 80 --normalize_features"
  # "python learned_osqp/train.py --precision high --types control --sizes 80 --adaptive_rho false --normalize_features"

  # "python learned_osqp/train.py --precision high --types lasso --sizes 15 --normalize_features"
  # "python learned_osqp/train.py --precision high --types lasso --sizes 15 --adaptive_rho false --normalize_features"
  
  # "python learned_osqp/train.py --precision high --types portfolio --sizes 15 --normalize_features"
  # "python learned_osqp/train.py --precision high --types portfolio --sizes 15 --adaptive_rho false --normalize_features"
)

# compute zero-based index for the bash array
IDX=$((SLURM_ARRAY_TASK_ID-1))

# safety check
if [ $IDX -lt 0 ] || [ $IDX -ge ${#CMDS[@]} ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

CMD="${CMDS[$IDX]}"

echo "Job $SLURM_JOB_ID (array ${SLURM_ARRAY_TASK_ID}) starting on $(hostname) at $(date)"
echo "Running: $CMD"

# run the mapped command under srun so Slurm tracks usage
srun $CMD

echo "Job $SLURM_JOB_ID (array ${SLURM_ARRAY_TASK_ID}) finished at $(date)"