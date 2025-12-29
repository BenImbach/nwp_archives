# Archive2SMET Pipeline

Extract HRDPS GRIB2 files and generate station-based SMET time series for full seasons on Alliance Fir.

## Quick Start

```bash
# Setup (one-time)
mkdir -p ~/scratch/archive2smet
cp archive2smet/*.py archive2smet/*.sh archive2smet/stations.geojson ~/scratch/archive2smet/
chmod +x ~/scratch/archive2smet/*.sh
cd ~/scratch/archive2smet

# Setup Python virtual environment (one-time, on login node)
./setup_home_venv.sh

# Run pipeline
./submit_archive2smet.sh 2023 /path/to/stations.geojson def-phaegeli
```

## Input

**GeoJSON file** with stations:
- Required: `station_id` (or `id`) property
- Geometry: Point (lat/lon, EPSG:4326)
- Elevations extracted from DEM (not from GeoJSON)

**GRIB files**: `/project/6005576/data/nwp/hrdps/<season>/HRDPS_YYYYMMDD.grib2`

## Output

**Location**: `$SCRATCH/nwp_archives/smet/<season>/`

**Files**: One SMET file per station (`<station_id>.smet`)

**Format**: Hourly data for full season (Sept 1 - May 31)
- Fields: `timestamp DW ILWR ISWR PSUM RH TA VW`
- Missing values: -999
- Elevation from DEM

## Pipeline

**Two phases** (automatically chained):
1. **Extract**: Parallel array jobs process time chunks (~7 days each)
2. **Concat**: Single job merges chunks into final station files

**Time limits**: Extract jobs (3h), Concat job (1h)

## Configuration

**Environment variables** (optional, defaults shown):
```bash
export SEASON=2023
export GEOJSON_FILE=/path/to/stations.geojson
export GRIB_DIR=/project/6005576/data/nwp/hrdps/${SEASON}
export OUTPUT_DIR=$SCRATCH/nwp_archives/smet/${SEASON}
```

**DEM selection**: Automatic based on season (2013-2017: WEST, 2018-2023: OLD, 2025+: current)

## Monitoring

```bash
squeue -u $USER                                    # Job status
tail -f $SCRATCH/nwp_archives/smet/2023/logs/*.log  # Logs
```

## Files

**Local directory**: `~/scratch/archive2smet/`
- `archive2smet.py` - Python code
- `archive2smet.sh` - Slurm job script
- `submit_archive2smet.sh` - Submission wrapper
- `setup_home_venv.sh` - Virtual environment setup (run once)
- `stations.geojson` - Example (optional)

## Notes

- Raw HRDPS values (no lapse rate corrections)
- Nearest-neighbor grid lookup
- Missing dates filled with -999
- Python packages installed in virtual environment (`$HOME/python/archive2smet`)
