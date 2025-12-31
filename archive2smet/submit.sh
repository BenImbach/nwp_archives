#!/bin/bash
# Submission script for archive2smet
# Processes entire season in one job with multiprocessing

# Usage: ./submit.sh <season> <geojson_file> [account] [cpus]
# Example: ./submit.sh 2023 /path/to/stations.geojson def-phaegeli 16

if [ $# -lt 2 ]; then
    echo "Usage: $0 <season> <geojson_file> [account] [cpus]"
    echo "Example: $0 2023 /path/to/stations.geojson def-phaegeli 16"
    exit 1
fi

SEASON=$1
GEOJSON_FILE=$2
ACCOUNT=${3:-def-phaegeli}
CPUS=${4:-16}

SCRIPT_DIR=$HOME/scratch/archive2smet
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/output/${SEASON}}
GRIB_DIR=${GRIB_DIR:-/project/6005576/data/nwp/hrdps/${SEASON}}

# Calculate total days for time estimate (Sept 1 prev year to May 31 season year)
TOTAL_DAYS=$(python3 -c "from datetime import datetime; s=$SEASON; print((datetime(s,5,31) - datetime(s-1,9,1)).days + 1)")

echo "Archive2SMET Pipeline: Season $SEASON, $TOTAL_DAYS days"
echo "Using $CPUS CPUs for parallel processing"
echo ""

mkdir -p ${SCRIPT_DIR}/logs/${SEASON}
mkdir -p ${OUTPUT_DIR}

# Submit single job
echo "Submitting job..."
JOB_OUTPUT=$(sbatch \
    --account=$ACCOUNT \
    --job-name=archive2smet \
    --time=12:00:00 \
    --mem=16G \
    --cpus-per-task=${CPUS} \
    --output=${SCRIPT_DIR}/logs/${SEASON}/%j.out \
    --error=${SCRIPT_DIR}/logs/${SEASON}/%j.err \
    --export=ALL,SEASON=$SEASON,GEOJSON_FILE=$GEOJSON_FILE,GRIB_DIR=$GRIB_DIR,OUTPUT_DIR=$OUTPUT_DIR \
    ${SCRIPT_DIR}/archive2smet.sh)

JOB=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$JOB" ]; then
    echo "Error: Failed to submit job"
    echo "sbatch output: $JOB_OUTPUT"
    exit 1
fi

echo "Job ID: $JOB"
echo ""
echo "Monitor: squeue -u $USER"
echo "Logs: ${SCRIPT_DIR}/logs/${SEASON}/${JOB}.out"
echo "Output: ${OUTPUT_DIR}/"

