#!/bin/bash
#SBATCH --job-name=cs5624_datagen
#SBATCH --partition=a30_normal_q
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "============================================="
echo "CS 5624 — Dataset Generation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "============================================="

# Ensure unbuffered Python output
export PYTHONUNBUFFERED=1

# Navigate to project root
cd "${HOME}/cs5624"

# Generate the full benchmark dataset
python3 -m graphharness.build_data

echo ""
echo "============================================="
echo "Dataset generation complete."
echo "============================================="

# Print summary
echo ""
echo "Files generated:"
ls -lh data/*.jsonl data/manifest.json 2>/dev/null
echo ""
wc -l data/*.jsonl 2>/dev/null
