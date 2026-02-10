#!/bin/bash
#SBATCH --job-name=hydraulic_ale
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=32       
#SBATCH --time=04:00:00    
#SBATCH --partition=express3

# 1. Load FEniCS module (depends on your cluster)
module load fenics

mpirun -n 32 python3 solver.py
