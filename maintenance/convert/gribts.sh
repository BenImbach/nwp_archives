#!/bin/bash
#SBATCH --account=def-phaegeli
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024M
#SBATCH --job-name=gribts_%a
#SBATCH --output=gribts_%a_%j.out
#SBATCH --error=gribts_%a_%j.err
#SBATCH --array=1-8

# Run times
# 2013: 18 min
# 2018: estimate 12 h for big grid

# Set season
SEASON=2022

# Load required modules
module load r

# Run the R script with season argument
Rscript gribts.R $SEASON

# Merge all season logs into a single file
output_log_file="logs/season_${SEASON}.log"
rm -f $output_log_file
for chunk_log in logs/season_${SEASON}_chunk_*.log; do
    cat $chunk_log >> $output_log_file
done
