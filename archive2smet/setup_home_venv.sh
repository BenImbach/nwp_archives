#!/bin/bash
# One-time setup script to create Python virtual environment in HOME directory
# Run this once on a login node
# Following: https://docs.alliancecan.ca/wiki/Python

set -e

echo "========================================="
echo "Python Virtual Environment Setup"
echo "Following: https://docs.alliancecan.ca/wiki/Python"
echo "========================================="
echo ""

# Step 1: Load required modules
echo "Step 1: Loading required modules..."
module purge
module load python
module load scipy-stack  # Provides: NumPy, SciPy, Matplotlib, pandas, etc.
module load eccodes      # Required for GRIB file support (cfgrib dependency)
module load proj         # Required for PROJ data directory (geopandas/pyproj)

echo "  ✓ Loaded: python (default version)"
echo "  ✓ Loaded: scipy-stack (provides numpy, pandas, scipy, etc.)"
echo "  ✓ Loaded: eccodes (for GRIB file support)"
echo "  ✓ Loaded: proj (for PROJ data directory)"
echo ""

# Step 2: Set virtual environment location
VENV_DIR=$HOME/python/archive2smet
mkdir -p $(dirname $VENV_DIR)

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "WARNING: Virtual environment already exists at: $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Step 2: Creating virtual environment..."
echo "  Location: $VENV_DIR"
echo ""

# Step 3: Create virtual environment
virtualenv --no-download $VENV_DIR

echo "  ✓ Virtual environment created"
echo ""

# Step 4: Activate the virtual environment
echo "Step 4: Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "  ✓ Virtual environment activated"
echo ""

# Step 5: Upgrade pip
echo "Step 5: Upgrading pip..."
pip install --no-index --upgrade pip

echo "  ✓ pip upgraded"
echo ""

# Step 6: Install NumPy 1.x (avoids compatibility issues with NumPy 2.x from scipy-stack)
echo "Step 6: Installing NumPy 1.x..."
pip install --no-index 'numpy<2.0'
echo "  ✓ NumPy 1.x installed"
echo ""

# Step 7: Install required packages
echo "Step 7: Installing required packages..."
pip install --no-index xarray cfgrib geopandas
echo "  ✓ Packages installed"
echo ""

# Step 8: Verify installation
echo "Step 8: Verifying installation..."
python -c "import numpy as np; print(f'  ✓ numpy {np.__version__} (installed in venv)')" || echo "  ✗ numpy import failed"
python -c "import pandas as pd; print(f'  ✓ pandas {pd.__version__} (from scipy-stack)')" || echo "  ✗ pandas import failed"
python -c "import scipy; print(f'  ✓ scipy {scipy.__version__} (from scipy-stack)')" || echo "  ✗ scipy import failed"
python -c "import xarray as xr; print(f'  ✓ xarray {xr.__version__}')" || echo "  ✗ xarray import failed"
python -c "import cfgrib; print(f'  ✓ cfgrib {cfgrib.__version__}')" || echo "  ✗ cfgrib import failed"

python -c "import geopandas as gpd; print(f'  ✓ geopandas {gpd.__version__}')" 2>&1 | grep -v "UserWarning" || echo "  ✗ geopandas import failed"
python -c "import eccodes; print(f'  ✓ eccodes Python bindings available')" 2>/dev/null || echo "  ⚠ eccodes Python package not available (cfgrib uses eccodes library from module)"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To use this environment in your jobs:"
echo "  1. Load the same modules:"
echo "     module purge"
echo "     module load python scipy-stack eccodes proj"
echo ""
echo "  2. Activate the virtual environment:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  3. Your archive2smet.sh script handles this automatically"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""
echo ""

