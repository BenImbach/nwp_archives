#!/bin/bash
#SBATCH --account=def-<account>
#SBATCH --job-name=archive2smet

MODE=${1:-extract}

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

mkdir -p ${SCRIPT_DIR}/logs
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

