#!/bin/bash
# One-time setup script to create Python virtual environment in HOME directory
# Run this once on a login node to set up the environment
# Following Alliance Canada best practices: https://docs.alliancecan.ca/wiki/Python
#
# Why HOME directory?
# - Virtual environments in $HOME are persistent and recommended
# - DO NOT create venvs in $SCRATCH as they may get partially deleted
# - For Trillium specifically, it's recommended to create venv from login node in HOME
#   and source it in your job script

set -e  # Exit on any error

echo "========================================="
echo "Python Virtual Environment Setup"
echo "Following: https://docs.alliancecan.ca/wiki/Python"
echo "========================================="
echo ""

# Step 1: Load required modules (using default versions as requested)
# Note: Always load modules BEFORE activating virtual environment
echo "Step 1: Loading required modules..."
module purge
module load python
module load scipy-stack  # Provides: NumPy, SciPy, Matplotlib, pandas, etc.
module load eccodes      # Required for GRIB file support (cfgrib dependency)

echo "  ✓ Loaded: python (default version)"
echo "  ✓ Loaded: scipy-stack (provides numpy, pandas, scipy, etc.)"
echo "  ✓ Loaded: eccodes (for GRIB file support)"
echo ""

# Step 2: Set virtual environment location
# Location: $HOME/python/archive2smet (in HOME directory as recommended)
VENV_DIR=$HOME/python/archive2smet

# Create parent directory if it doesn't exist
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

# Step 3: Create virtual environment using --no-download flag
# The --no-download flag ensures we use Alliance Canada's pre-built wheels
# This is more reliable than downloading from PyPI
virtualenv --no-download $VENV_DIR

echo "  ✓ Virtual environment created"
echo ""

# Step 4: Activate the virtual environment
echo "Step 3: Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "  ✓ Virtual environment activated"
echo ""

# Step 5: Upgrade pip to latest version
# Important: Always upgrade pip first to ensure dependency resolution works correctly
# Use --no-index to get pip from Alliance Canada's wheels
echo "Step 4: Upgrading pip..."
pip install --no-index --upgrade pip

echo "  ✓ pip upgraded"
echo ""

# Step 6: Install NumPy 1.x to avoid compatibility issues
# The scipy-stack module provides NumPy 2.x, but many packages (numexpr, cftime, etc.)
# were compiled with NumPy 1.x and are incompatible. Installing NumPy 1.x in the venv
# will override the module's NumPy 2.x for packages in the venv.
echo "Step 5: Installing NumPy 1.x for compatibility..."
echo "  (This overrides NumPy 2.x from scipy-stack module to avoid compatibility issues)"
pip install --no-index 'numpy<2.0'

echo "  ✓ NumPy 1.x installed"
echo ""

# Step 7: Install required packages
# Using --no-index flag to use only Alliance Canada's pre-built wheels
# Installing packages together helps pip resolve dependencies
# 
# Required packages for archive2smet.py:
# - xarray: for working with GRIB/netCDF data
# - cfgrib: GRIB file backend for xarray (requires eccodes module)
# - geopandas: for reading GeoJSON station files
# 
# Note: pandas, scipy, numpy are provided by scipy-stack module and
#       don't need to be installed in the venv (but will be available)
echo "Step 6: Installing required packages from Alliance Canada wheels..."
echo "  Checking available wheels..."

# Check if packages are available (optional but helpful)
if command -v avail_wheels &> /dev/null; then
    echo "  Available wheels:"
    avail_wheels xarray cfgrib geopandas 2>/dev/null || echo "    (avail_wheels not available, continuing anyway)"
    echo ""
fi

# Install packages together to help pip resolve dependencies
echo "  Installing: xarray cfgrib geopandas"
pip install --no-index xarray cfgrib geopandas

echo "  ✓ Packages installed"
echo ""

# Step 8: Verify installation
echo "Step 7: Verifying installation..."
python -c "import numpy as np; print(f'  ✓ numpy {np.__version__} (installed in venv)')" || echo "  ✗ numpy import failed"
python -c "import pandas as pd; print(f'  ✓ pandas {pd.__version__} (from scipy-stack)')" || echo "  ✗ pandas import failed"
python -c "import scipy; print(f'  ✓ scipy {scipy.__version__} (from scipy-stack)')" || echo "  ✗ scipy import failed"
python -c "import xarray as xr; print(f'  ✓ xarray {xr.__version__}')" || echo "  ✗ xarray import failed"
python -c "import cfgrib; print(f'  ✓ cfgrib {cfgrib.__version__}')" || echo "  ✗ cfgrib import failed"
python -c "import geopandas as gpd; print(f'  ✓ geopandas {gpd.__version__}')" || echo "  ✗ geopandas import failed"
# Check if eccodes Python package is available (provided by eccodes module)
python -c "import eccodes; print(f'  ✓ eccodes Python bindings available')" 2>/dev/null || echo "  ⚠ eccodes Python package not available (cfgrib uses eccodes library from module)"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To use this environment in your jobs:"
echo "  1. Load the same modules:"
echo "     module purge"
echo "     module load python scipy-stack eccodes"
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

