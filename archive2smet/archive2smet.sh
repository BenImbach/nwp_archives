#!/bin/bash
#SBATCH --account=def-<account>
#SBATCH --job-name=archive2smet

SEASON=${SEASON:-2023}
GEOJSON_FILE=${GEOJSON_FILE:-/path/to/stations.geojson}
GRIB_DIR=${GRIB_DIR:-/project/6005576/data/nwp/hrdps/${SEASON}}

SCRIPT_DIR=$HOME/scratch/archive2smet

# Change to script directory to ensure relative paths work
cd "${SCRIPT_DIR}" || {
    echo "Error: Cannot change to script directory: ${SCRIPT_DIR}"
    echo "Current directory: $(pwd)"
    exit 1
}

OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/output/${SEASON}}
mkdir -p ${OUTPUT_DIR}

# Load modules
module purge
module load python scipy-stack eccodes proj

# Activate virtual environment
VENV_DIR=$HOME/python/archive2smet

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run setup_home_venv.sh first to create the environment"
    exit 1
fi

source $VENV_DIR/bin/activate

# Set log file location
LOG_DIR=${SCRIPT_DIR}/logs/${SEASON}
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/process.log

# Run processing
python archive2smet.py --season $SEASON --geojson $GEOJSON_FILE \
    --grib-dir $GRIB_DIR --output-dir $OUTPUT_DIR --log-file $LOG_FILE
