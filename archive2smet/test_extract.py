#!/usr/bin/env python3
"""
Interactive test script to extract GRIB data for a single file and point
Follows the same extraction logic as archive2smet.py

Usage:
    module purge
    module load python scipy-stack eccodes proj
    source $HOME/python/archive2smet/bin/activate
    python
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree
from datetime import datetime
import tempfile
import shutil

# Configuration
GRIB_FILE = '/project/6005576/data/nwp/hrdps/2025/HRDPS_20250101.grib2'
TEST_LON = -115.0
TEST_LAT = 50.0
TEST_ID = 'test_point'

# Constants (from archive2smet.py)
KELVIN_TO_CELSIUS = 273.15
TEMP_KELVIN_THRESHOLD = 100

variable_mapping = {
    't': 'TA', 't2m': 'TA', 'r': 'RH', 'r2': 'RH', '2r': 'RH',
    'ws': 'VW', 'wdir': 'DW', 'wdir10': 'DW',
    'u10': 'VW_U', 'v10': 'VW_V', 'si10': 'VW_ALT',
    'sdswrf': 'ISWR', 'sdlwrf': 'ILWR', 'dswrf': 'ISWR', 'dlwrf': 'ILWR',
    'unknown': 'PSUM', 'tp': 'PSUM', 'prate': 'PSUM',
}

# ============================================================================
# STEP 1: Setup temp directory and open GRIB file
# ============================================================================

temp_dir = tempfile.mkdtemp()
index_path = os.path.join(temp_dir, Path(GRIB_FILE).name + '.idx')

datasets = []
try:
    datasets.append(xr.open_dataset(GRIB_FILE, engine='cfgrib', 
                                   backend_kwargs={'indexpath': index_path}))
    print(f"✓ Opened file with {len(datasets[0].data_vars)} variables")
except Exception as e:
    print(f"Trying with level filters: {e}")
    for level in ['surface', 'heightAboveGround', 'atmosphere']:
        try:
            level_path = os.path.join(temp_dir, f"{Path(GRIB_FILE).stem}.{level}.idx")
            ds = xr.open_dataset(GRIB_FILE, engine='cfgrib',
                               backend_kwargs={'filter_by_keys': {'typeOfLevel': level},
                                              'indexpath': level_path})
            datasets.append(ds)
            print(f"✓ Opened {level} level")
        except:
            pass

if not datasets:
    raise ValueError(f"Could not open {GRIB_FILE}")

# ============================================================================
# STEP 2: Extract data for single point
# ============================================================================

all_data = []
for ds in datasets:
    available_vars = [v for v in ds.data_vars if v in variable_mapping]
    if not available_vars:
        continue
    
    lon_coord = 'longitude' if 'longitude' in ds.coords else 'x'
    lat_coord = 'latitude' if 'latitude' in ds.coords else 'y'
    time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'
    
    lon_dims, lat_dims = ds.coords[lon_coord].dims, ds.coords[lat_coord].dims
    is_2d = len(lon_dims) == 2 and len(lat_dims) == 2
    
    if is_2d:
        grid_coords = np.column_stack([ds.coords[lon_coord].values.flatten(),
                                      ds.coords[lat_coord].values.flatten()])
        _, indices = cKDTree(grid_coords).query([[TEST_LON, TEST_LAT]])
        y_size, x_size = ds.coords[lon_coord].shape
        y_indices, x_indices = indices // x_size, indices % x_size
    else:
        y_indices, x_indices = None, None
    
    for var in available_vars:
        try:
            if is_2d:
                var_data = ds[var].isel(y=xr.DataArray(y_indices, dims='points'),
                                       x=xr.DataArray(x_indices, dims='points'))
            else:
                var_data = ds[var].sel({lon_coord: TEST_LON, 
                                       lat_coord: TEST_LAT}, method='nearest')
            
            df = var_data.to_dataframe(name='value').reset_index()
            df['variable'] = variable_mapping[var]
            df['ID'] = TEST_ID
            df['timestamp'] = pd.to_datetime(df[time_coord])
            all_data.append(df[['ID', 'timestamp', 'variable', 'value']])
            print(f"  ✓ Extracted {var} -> {variable_mapping[var]}")
        except Exception as e:
            print(f"  ✗ Failed {var}: {e}")

print(f"\nExtracted {len(all_data)} variable datasets")

# ============================================================================
# STEP 3: Combine and handle wind components
# ============================================================================

if all_data:
    df_long = pd.concat(all_data, ignore_index=True)
    print(f"\nLong format: {len(df_long)} rows")
    print(df_long.head(10))
    
    # Handle wind components (u10, v10 -> VW, DW)
    if 'VW_U' in df_long['variable'].values and 'VW_V' in df_long['variable'].values:
        wind_df = df_long[df_long['variable'].isin(['VW_U', 'VW_V'])].pivot_table(
            index=['ID', 'timestamp'], columns='variable', values='value', aggfunc='first').reset_index()
        wind_df['VW'] = np.sqrt(wind_df['VW_U']**2 + wind_df['VW_V']**2)
        wind_df['DW'] = (270 - np.arctan2(wind_df['VW_V'], wind_df['VW_U']) * 180 / np.pi) % 360
        wind_long = wind_df[['ID', 'timestamp', 'VW', 'DW']].melt(
            id_vars=['ID', 'timestamp'], var_name='variable', value_name='value')
        df_long = pd.concat([df_long[~df_long['variable'].isin(['VW_U', 'VW_V'])], wind_long], ignore_index=True)
        print(f"\n✓ Converted wind components to VW/DW")

# ============================================================================
# STEP 4: Reshape to wide format and convert temperature
# ============================================================================

if 'df_long' in locals():
    df_wide = df_long.pivot_table(index=['ID', 'timestamp'], columns='variable', values='value', 
                                 aggfunc='first').reset_index()
    
    print(f"\nWide format columns: {list(df_wide.columns)}")
    print(df_wide.head())
    
    # Convert temperature from Kelvin if needed
    if 'TA' in df_wide.columns and df_wide['TA'].mean() > TEMP_KELVIN_THRESHOLD:
        df_wide['TA'] = df_wide['TA'] - KELVIN_TO_CELSIUS
        print(f"\n✓ Converted temperature from Kelvin to Celsius")
        print(f"  TA range: {df_wide['TA'].min():.1f} to {df_wide['TA'].max():.1f} °C")
    
    print(f"\nFinal data:")
    print(df_wide.head(10))

# ============================================================================
# CLEANUP
# ============================================================================
"""
shutil.rmtree(temp_dir, ignore_errors=True)
print(f"\\n✓ Cleaned up {temp_dir}")
"""
