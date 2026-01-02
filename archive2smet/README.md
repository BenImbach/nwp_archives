# Archive2SMET Pipeline

Extract HRDPS GRIB2 files and generate station-based SMET time series for full seasons on Alliance Canada systems.

## Quick Start

```bash
# Setup (one-time)
cd ~/scratch
git clone https://bitbucket.org/sfu-arp/nwp_archives.git
cd nwp_archives/archive2smet
chmod +x *.sh

# Setup Python virtual environment (one-time, on login node)
./setup_home_venv.sh

# Run pipeline
./submit.sh 2025 stations.geojson 16
```

## Input

**GeoJSON file** with stations:

- Required: `station_id` (or `id`) property
- Geometry: Point (lat/lon, EPSG:4326)
- Elevations extracted from DEM (not from GeoJSON)

**GRIB files**: `/project/6005576/data/nwp/hrdps/<season>/HRDPS_YYYYMMDD.grib2`

## Output

**Location**: `~/scratch/nwp_archives/output/<season>/`

**Files**: One SMET file per station (`<station_id>.smet`)

**Format**: Hourly data for full season (Sept 1 prev year - May 31 season year)

- Fields: `timestamp DW ILWR ISWR PSUM RH TA VW`
- Missing values: -999
- Elevation from DEM based on season:
  - 2013-2017: HRDPS_WEST
  - 2018-2023: HRDPS_OLD
  - 2025+: HRDPS

## Pipeline

- Single job processes all stations for entire season
- Extracts all GRIB files in parallel using multiprocessing
- Uses configurable number of CPUs (default: 16)
- Time limit: 3 hours (adjustable in `submit.sh`)

## Configuration

**Command line arguments:**
```bash
./submit.sh <season> <geojson_file> [cpus]
```

The account/group is automatically detected from your session (first group starting with "def-").

## Monitoring

```bash
squeue -u $USER                                    # Job status
tail -f ~/scratch/nwp_archives/logs/2025/2025_stations_*.out    # SLURM job logs
```

Log files are named: `<season>_<stations>_<job_id>.out` where `<stations>` is the basename of your GeoJSON file.

## Requirements

**Modules** (loaded automatically by scripts):

- `python` - Python interpreter
- `scipy-stack` - Scientific Python packages (numpy, pandas, scipy)
- `eccodes` - GRIB file support
- `proj` - Geospatial coordinate transformations

**Python packages** (installed in venv):

- `numpy<2.0` - Override NumPy 2.x from scipy-stack for compatibility
- `xarray` - GRIB/netCDF data handling
- `cfgrib` - GRIB backend for xarray
- `geopandas` - GeoJSON file reading

**Setup**: Virtual environment created at `$HOME/python/archive2smet` (one-time setup via `setup_home_venv.sh`)
