#!/bin/bash
# Wrapper script to submit both phases automatically

# Usage: ./submit_archive2smet.sh <season> <geojson_file> [account]
# Example: ./submit_archive2smet.sh 2023 /path/to/stations.geojson def-phaegeli

if [ $# -lt 2 ]; then
    echo "Usage: $0 <season> <geojson_file> [account]"
    echo "Example: $0 2023 /path/to/stations.geojson def-phaegeli"
    exit 1
fi

SEASON=$1
GEOJSON_FILE=$2
ACCOUNT=${3:-def-phaegeli}

# Script directory (where scripts are located)
# Scripts are always in $HOME/scratch/archive2smet
SCRIPT_DIR=$HOME/scratch/archive2smet

# Set output directory (in same folder as scripts, in output subdirectory)
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/output/${SEASON}}
GRIB_DIR=${GRIB_DIR:-/project/6005576/data/nwp/hrdps/${SEASON}}

# Export variables for both jobs
export SEASON=$SEASON
export GEOJSON_FILE=$GEOJSON_FILE
export GRIB_DIR=$GRIB_DIR
export OUTPUT_DIR=$OUTPUT_DIR

# Calculate total chunks dynamically based on season dates
# Season: Sept 1 (prev year) to May 31 (season year) with 7 days per chunk
# Use Python to calculate the same way archive2smet.py does it
TOTAL_CHUNKS=$(python3 -c "
from datetime import datetime
season = $SEASON
start_date = datetime(season - 1, 9, 1)
end_date = datetime(season, 5, 31)
total_days = (end_date - start_date).days + 1
days_per_chunk = 7
total_chunks = int((total_days + days_per_chunk - 1) / days_per_chunk)
print(total_chunks)
")

echo "Archive2SMET Pipeline: Season $SEASON, $TOTAL_CHUNKS chunks"
echo ""

# Create directories (logs and output are separate)
mkdir -p ${SCRIPT_DIR}/logs
mkdir -p ${OUTPUT_DIR}

# Submit extraction array job
echo "Submitting extraction array job..."
EXTRACT_JOB_OUTPUT=$(sbatch \
    --account=$ACCOUNT \
    --array=1-${TOTAL_CHUNKS} \
    --time=3:00:00 \
    --mem=8G \
    --cpus-per-task=4 \
    --output=${SCRIPT_DIR}/logs/%j_%a.out \
    --error=${SCRIPT_DIR}/logs/%j_%a.err \
    --export=ALL \
    ${SCRIPT_DIR}/archive2smet.sh extract)
EXTRACT_JOB=$(echo "$EXTRACT_JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$EXTRACT_JOB" ]; then
    echo "Error: Failed to submit extraction job"
    exit 1
fi

echo "Extraction job ID: $EXTRACT_JOB"
echo ""

# Submit concatenation job (runs automatically after all extract jobs complete)
echo "Submitting concatenation job (runs after all extract jobs complete)..."
# Wait a moment for job to be fully registered
sleep 1
# For array jobs, afterok waits for ALL array tasks to complete
CONCAT_JOB_OUTPUT=$(sbatch \
    --account=$ACCOUNT \
    --dependency=afterok:${EXTRACT_JOB} \
    --time=1:00:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --output=${SCRIPT_DIR}/logs/%j.out \
    --error=${SCRIPT_DIR}/logs/%j.err \
    --export=ALL \
    ${SCRIPT_DIR}/archive2smet.sh concat 2>&1)
CONCAT_JOB=$(echo "$CONCAT_JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

if [ -z "$CONCAT_JOB" ]; then
    echo "Error: Failed to submit concatenation job"
    echo "sbatch output: $CONCAT_JOB_OUTPUT"
    exit 1
fi

echo "Concatenation job ID: $CONCAT_JOB"
echo ""
echo "Jobs submitted: Extract=$EXTRACT_JOB (array), Concat=$CONCAT_JOB (auto-runs after extract)"
echo "Monitor: squeue -u $USER"
echo "Logs: ${SCRIPT_DIR}/logs/"
echo "Output: ${OUTPUT_DIR}/"

