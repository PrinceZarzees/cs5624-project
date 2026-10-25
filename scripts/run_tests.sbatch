#!/bin/bash
#SBATCH --job-name=cs5624_tests
#SBATCH --partition=a30_normal_q
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "============================================="
echo "CS 5624 — Running pytest Test Suite"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "============================================="

# Navigate to project root
cd "${HOME}/cs5624"

# Ensure pytest is on PATH
export PATH="${HOME}/.local/bin:${PATH}"

# Run tests with verbose output
python3 -m pytest tests/ -v --tb=long 2>&1

echo ""
echo "============================================="
echo "Test suite complete."
echo "============================================="
