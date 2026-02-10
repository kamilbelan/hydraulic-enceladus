#!/bin/bash -l
#SBATCH --job-name=enceladus_ale
#SBATCH --output=logs/run.%j.out
#SBATCH --error=logs/run.%j.err
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --partition=express3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cluster@kamilbelan.anonaddy.com


# ==========================================
# DIRECTORY SETUP
# ==========================================

PROJECT_ROOT=$(pwd)
RUN_ID="job_${SLURM_JOB_ID}"
OUT_DIR="${PROJECT_ROOT}/data/${RUN_ID}"
mkdir -p "$OUT_DIR"

# create a symlink to the latest run; convenience reaons
rm -f "${PROJECT_ROOT}/data/latest"
ln -s "$OUT_DIR" "${PROJECT_ROOT}/data/latest"

# ==========================================
# RUN 
# ==========================================

echo "========================================"
echo " Starting Job: $RUN_ID"
echo " Output Dir:   $OUT_DIR"
echo "========================================"


# module loading
module load fenics/master

# pipe python stdout/stderr to a log file inside the folder
srun python -u src/main.py --outdir "$OUT_DIR" > "$OUT_DIR/simulation.log" 2>&1

echo "Job finished. Check $OUT_DIR/simulation.log for details."
