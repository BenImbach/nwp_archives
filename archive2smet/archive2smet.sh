#!/bin/bash
#SBATCH --account=def-<account>
#SBATCH --job-name=archive2smet

MODE=${1:-extract}

SEASON=${SEASON:-2023}
GEOJSON_FILE=${GEOJSON_FILE:-/path/to/stations.geojson}
GRIB_DIR=${GRIB_DIR:-/project/6005576/data/nwp/hrdps/${SEASON}}

# Script directory (where this script and archive2smet.py are located)
# Scripts are always in $HOME/scratch/archive2smet
SCRIPT_DIR=$HOME/scratch/archive2smet

# Change to script directory to ensure relative paths work
cd "${SCRIPT_DIR}" || {
    echo "Error: Cannot change to script directory: ${SCRIPT_DIR}"
    echo "Current directory: $(pwd)"
    exit 1
}

# Output directory (in same folder as scripts, in output subdirectory)
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/output/${SEASON}}

# Create directories (logs and output are separate)
# Note: Logs directory should already exist (created by submit script)
# but create it here as well just in case
mkdir -p ${SCRIPT_DIR}/logs
mkdir -p ${OUTPUT_DIR}

# Load modules (use default versions)
# Following: https://docs.alliancecan.ca/wiki/Python
module purge
module load python
module load scipy-stack
module load eccodes

# Activate virtual environment (created by setup_home_venv.sh)
# Virtual environment location in HOME directory as recommended by Alliance Canada
# See: https://docs.alliancecan.ca/wiki/Python
VENV_DIR=$HOME/python/archive2smet

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run setup_home_venv.sh first to create the environment"
    exit 1
fi

source $VENV_DIR/bin/activate

if [ "$MODE" == "extract" ]; then
    [ -z "$SLURM_ARRAY_TASK_ID" ] && echo "Error: SLURM_ARRAY_TASK_ID not set" && exit 1
    python archive2smet.py extract --season $SEASON --chunk-id ${SLURM_ARRAY_TASK_ID} \
        --geojson $GEOJSON_FILE --grib-dir $GRIB_DIR --output-dir $OUTPUT_DIR
elif [ "$MODE" == "concat" ]; then
    python archive2smet.py concat --season $SEASON --geojson $GEOJSON_FILE \
        --chunk-dir $OUTPUT_DIR/chunks --output-dir $OUTPUT_DIR
else
    echo "Error: MODE must be 'extract' or 'concat'" && exit 1
fi

