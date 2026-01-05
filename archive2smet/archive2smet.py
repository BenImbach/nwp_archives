#!/usr/bin/env python3
"""
Archive2SMET Pipeline

Extracts meteorological data from HRDPS GRIB2 files at station locations
and generates SMET format time series files for SNOWPACK.

Main workflow:
1. Load stations from GeoJSON
2. Extract elevations from DEM
3. Extract meteorological data from GRIB files (parallel processing)
4. Combine and write SMET files

Usage:
    python archive2smet.py --season 2023 --geojson stations.geojson \\
                            --grib-dir /path/to/grib --output-dir /path/to/output
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
from pathlib import Path
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import tempfile
import shutil
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# Constants
KELVIN_TO_CELSIUS = 273.15
TEMP_KELVIN_THRESHOLD = 100  # If mean temp > 100, assume Kelvin
MISSING_VALUE = -999
REQUIRED_SMET_COLS = ['DW', 'ILWR', 'ISWR', 'PSUM', 'RH', 'TA', 'VW']

VARIABLE_MAPPING = {
    't': 'TA', 't2m': 'TA', 'r': 'RH', 'r2': 'RH', '2r': 'RH',
    'ws': 'VW', 'wdir': 'DW', 'wdir10': 'DW',
    'u10': 'VW_U', 'v10': 'VW_V', 'si10': 'VW_ALT',
    'sdswrf': 'ISWR', 'sdlwrf': 'ILWR', 'dswrf': 'ISWR', 'dlwrf': 'ILWR',
    'unknown': 'PSUM', 'tp': 'PSUM', 'prate': 'PSUM',
}


# ============================================================================
# Date/Season Utilities
# ============================================================================

def get_season_dates(season):
    """Get start (Sept 1 prev year) and end (May 31 season year) dates"""
    return datetime(int(season) - 1, 9, 1), datetime(int(season), 5, 31)


def get_dem_file(season, dem_base_dir="/project/6005576/data/nwp/dem"):
    """Get DEM file path based on season"""
    season_int = int(season)
    if season_int <= 2017:
        return Path(dem_base_dir) / "HRDPS_WEST_DEM.grib2"
    elif season_int <= 2023:
        return Path(dem_base_dir) / "HRDPS_OLD_DEM.grib2"
    else:
        return Path(dem_base_dir) / "HRDPS_DEM.grib2"


def ensure_station_id(stations):
    """Ensure stations have station_id column"""
    if 'station_id' not in stations.columns:
        stations['station_id'] = stations.get('id', [f"station_{i}" for i in range(len(stations))])
    return stations


def log_message(message, level="INFO"):
    """Log message with timestamp (outputs to stdout/stderr, captured by SLURM)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | {level} | {message}"
    if level == "ERROR":
        print(log_entry, file=sys.stderr)
    else:
        print(log_entry)


# ============================================================================
# Helper Functions
# ============================================================================

def _get_coords_info(ds):
    """Get coordinate names and check if 2D"""
    lon_coord = 'longitude' if 'longitude' in ds.coords else 'x'
    lat_coord = 'latitude' if 'latitude' in ds.coords else 'y'
    lon_dims, lat_dims = ds.coords[lon_coord].dims, ds.coords[lat_coord].dims
    is_2d = len(lon_dims) == 2 and len(lat_dims) == 2
    return lon_coord, lat_coord, is_2d


def _normalize_longitude(lons):
    """Normalize longitudes to 0-360 convention"""
    lons = np.array(lons).copy()
    lons[lons < 0] += 360
    return lons


def _find_nearest_grid_points(grid_lons, grid_lats, query_lons, query_lats):
    """Find nearest grid points for query locations using cKDTree.
    
    Returns:
        flat_indices: Flattened indices into grid arrays
        distances: Distances in degrees
    """
    grid_coords = np.column_stack([_normalize_longitude(grid_lons), grid_lats])
    query_points = np.column_stack([_normalize_longitude(query_lons), query_lats])
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)
    
    distances, flat_indices = cKDTree(grid_coords).query(query_points)
    flat_indices = np.atleast_1d(flat_indices).astype(int)
    
    if np.any(distances > 1.0):
        print(f"  Warning: Some points are far from grid (max distance: {distances.max():.3f} degrees)")
    
    return flat_indices, distances


def _preserve_coordinates(var_data, var_dims, new_dims, n_points):
    """Preserve only essential coordinates when reshaping DataArray.
    
    Keeps non-spatial dimension coordinates and valid_time (most important).
    """
    # Keep coordinates for non-spatial dimensions
    new_coords = {dim: var_data.coords[dim] for dim in var_dims if dim not in ['y', 'x']}
    
    # Only preserve valid_time if it exists (essential for timestamp extraction)
    if 'valid_time' in var_data.coords:
        coord_dims = var_data.coords['valid_time'].dims
        if all(dim in new_dims for dim in coord_dims):
            new_coords['valid_time'] = var_data.coords['valid_time']
    
    new_coords['points'] = np.arange(n_points)
    return new_coords


def _extract_at_points(ds, var, lons, lats, lon_coord, lat_coord, is_2d):
    """Extract variable data at point locations"""
    if is_2d:
        # Find nearest grid points
        grid_lons = ds.coords[lon_coord].values.flatten()
        grid_lats = ds.coords[lat_coord].values.flatten()
        flat_indices, distances = _find_nearest_grid_points(grid_lons, grid_lats, lons, lats)
        
        # Convert to (y, x) indices
        y_size, x_size = ds.coords[lon_coord].shape
        y_indices = flat_indices // x_size
        x_indices = flat_indices % x_size
        
        # Extract data using vectorized indexing (MUCH faster than loop)
        var_data = ds[var]
        var_dims = list(var_data.dims)
        data_array = var_data.values
        
        # Build index arrays for all points at once
        index_list = []
        for dim in var_dims:
            if dim == 'y':
                index_list.append(y_indices)
            elif dim == 'x':
                index_list.append(x_indices)
            else:
                index_list.append(slice(None))
        
        # Extract all points at once using advanced indexing
        values = data_array[tuple(index_list)]
        
        # Debug first point if all NaN
        if len(lons) > 0:
            first_point_data = values[..., 0] if values.ndim > 1 else values[0]
            if np.all(np.isnan(first_point_data)):
                print(f"  Warning: Point 0 at ({lons[0]:.3f}, {lats[0]:.3f}) -> grid ({y_indices[0]}, {x_indices[0]}) returned all NaN")
        
        # Transpose to get (non_spatial_dims, points) format
        if values.ndim > 1:
            # Move first dimension (points) to last position
            values = np.moveaxis(values, 0, -1)
        new_dims = [d if d not in ['y', 'x'] else 'points' for d in var_dims]
        new_dims = [d for i, d in enumerate(new_dims) if d != 'points' or i == new_dims.index('points')]
        new_coords = _preserve_coordinates(var_data, var_dims, new_dims, len(lons))
        
        return xr.DataArray(values, dims=new_dims, coords=new_coords, attrs=var_data.attrs)
    else:
        # For 1D coordinates, use xarray's built-in selection
        return ds[var].sel({lon_coord: xr.DataArray(lons, dims='points'),
                           lat_coord: xr.DataArray(lats, dims='points')}, method='nearest')


def _open_grib_datasets(grib_file, temp_dir):
    """Open GRIB file, trying different methods (optimized order)"""
    index_path = os.path.join(temp_dir, Path(grib_file).name + '.idx')
    
    # First try cfgrib.open_datasets() - handles step conflicts (most common case)
    try:
        datasets = cfgrib.open_datasets(grib_file, backend_kwargs={'indexpath': index_path})
        if datasets:
            return datasets
    except Exception:
        pass
    
    # Fallback: direct open (faster for files without conflicts)
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib', 
                             backend_kwargs={'indexpath': index_path})
        if len(ds.data_vars) > 0:
            return [ds]
    except Exception:
        pass
    
    # Last resort: level filters (only if above methods fail)
    datasets = []
    for level in ['surface', 'heightAboveGround', 'atmosphere']:
        try:
            level_path = os.path.join(temp_dir, f"{Path(grib_file).stem}.{level}.idx")
            ds = xr.open_dataset(grib_file, engine='cfgrib',
                                backend_kwargs={'filter_by_keys': {'typeOfLevel': level},
                                               'indexpath': level_path})
            if len(ds.data_vars) > 0:
                datasets.append(ds)
        except Exception:
            pass
    
    return datasets if datasets else []


# ============================================================================
# DEM Elevation Extraction
# ============================================================================

def extract_station_elevations(dem_file, stations):
    """Extract elevation from DEM at station locations"""
    print(f"Extracting elevations from {dem_file} for {len(stations)} stations")
    
    # Open DEM
    try:
        ds = xr.open_dataset(dem_file, engine='cfgrib')
    except:
        ds = xr.open_dataset(dem_file, engine='cfgrib', 
                            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    
    # Find elevation variable
    elev_var = next((v for v in ds.data_vars if v.lower() in ['z', 'orog', 'elevation', 'height', 'hgt']), 
                    list(ds.data_vars)[0])
    
    # Extract coordinates
    lons, lats = stations.geometry.x.values, stations.geometry.y.values
    station_ids = stations.get('station_id', [f"station_{i}" for i in range(len(stations))]).values
    lon_coord, lat_coord, is_2d = _get_coords_info(ds)
    
    # Extract elevations
    elev_data = _extract_at_points(ds, elev_var, lons, lats, lon_coord, lat_coord, is_2d)
    elevations = elev_data.values
    print(f"  Extracted elevations: {min(elevations):.1f} - {max(elevations):.1f} m")
    return {sid: float(elev) for sid, elev in zip(station_ids, elevations)}


# ============================================================================
# GRIB Extraction
# ============================================================================

def _extract_timestamp(df, var_data, ds):
    """Extract timestamp from extracted variable data.
    
    Handles different GRIB time coordinate conventions:
    - valid_time coordinate (preferred)
    - time dimension
    - step dimension with base time
    """
    # Prefer valid_time coordinate (most common in HRDPS files)
    if 'valid_time' in var_data.coords:
        if 'step' in df.columns:
            # Map step values to valid_time
            step_to_time = pd.Series(
                pd.to_datetime(var_data.coords['valid_time'].values),
                index=var_data.coords['step'].values
            )
            return df['step'].map(step_to_time)
        elif 'valid_time' in df.columns:
            return pd.to_datetime(df['valid_time'])
        else:
            return pd.to_datetime(var_data.coords['valid_time'].values)
    
    # Fallback to time dimension
    if 'time' in df.columns:
        return pd.to_datetime(df['time'])
    
    # Fallback to step + base time
    if 'step' in df.columns:
        if 'time' in ds.coords:
            return pd.to_datetime(ds.coords['time'].values) + pd.to_timedelta(df['step'])
        raise ValueError("Cannot determine timestamp: have 'step' but no 'time' coordinate")
    
    raise ValueError(f"Cannot determine timestamp. Columns: {df.columns.tolist()}, coords: {list(var_data.coords.keys())}")


def _extract_variable_data(ds, var, lons, lats, point_ids, lon_coord, lat_coord, is_2d):
    """Extract data for a single variable and return as long-format DataFrame.
    
    Returns DataFrame with columns: ID, timestamp, variable, value
    """
    var_data = _extract_at_points(ds, var, lons, lats, lon_coord, lat_coord, is_2d)
    df = var_data.to_dataframe(name='value').reset_index()
    
    # Extract timestamp
    df['timestamp'] = _extract_timestamp(df, var_data, ds)
    
    # Map GRIB variable name to SMET variable name
    df['variable'] = VARIABLE_MAPPING[var]
    
    # Assign station IDs based on points dimension
    if 'points' in df.columns:
        df['ID'] = df['points'].map(lambda idx: point_ids[idx])
    else:
        # If no points dimension, tile station IDs
        n_timesteps = len(df) // len(point_ids)
        df['ID'] = np.tile(point_ids, n_timesteps)[:len(df)]
    
    return df[['ID', 'timestamp', 'variable', 'value']]


def _process_wind_components(df_long):
    """Convert u/v wind components to speed and direction.
    
    If u10/v10 components exist, calculates:
    - VW: wind speed (m/s)
    - DW: wind direction (degrees, 0-360, 0=North)
    """
    has_u = 'VW_U' in df_long['variable'].values
    has_v = 'VW_V' in df_long['variable'].values
    if not (has_u and has_v):
        return df_long
    
    # Pivot to get u and v in same rows
    wind_df = df_long[df_long['variable'].isin(['VW_U', 'VW_V'])].pivot_table(
        index=['ID', 'timestamp'], columns='variable', values='value', aggfunc='first'
    ).reset_index()
    
    # Calculate speed and direction
    wind_df['VW'] = np.sqrt(wind_df['VW_U']**2 + wind_df['VW_V']**2)
    # Direction: 270 - atan2(v, u) converts from meteorological to compass convention
    wind_df['DW'] = (270 - np.arctan2(wind_df['VW_V'], wind_df['VW_U']) * 180 / np.pi) % 360
    
    # Convert back to long format
    wind_long = wind_df[['ID', 'timestamp', 'VW', 'DW']].melt(
        id_vars=['ID', 'timestamp'], var_name='variable', value_name='value'
    )
    
    # Replace u/v components with speed/direction
    return pd.concat([df_long[~df_long['variable'].isin(['VW_U', 'VW_V'])], wind_long], ignore_index=True)


def extract_grib_file(grib_file, stations, point_id_col="station_id"):
    """Extract GRIB data at station locations and return as wide-format DataFrame.
    
    Args:
        grib_file: Path to GRIB2 file
        stations: GeoDataFrame with station locations
        point_id_col: Column name for station IDs
    
    Returns:
        DataFrame with columns: ID, timestamp, and all SMET variables
    """
    print(f"Extracting {len(stations)} points from {grib_file}")
    start_time = datetime.now()
    
    temp_dir = tempfile.mkdtemp()
    try:
        datasets = _open_grib_datasets(grib_file, temp_dir)
        if not datasets:
            raise ValueError(f"Could not open {grib_file}")
        
        lons, lats = stations.geometry.x.values, stations.geometry.y.values
        point_ids = stations[point_id_col].values
        
        # Extract all variables from all datasets
        all_data = []
        for ds in datasets:
            available_vars = [v for v in ds.data_vars if v in VARIABLE_MAPPING]
            if not available_vars:
                continue
            
            lon_coord, lat_coord, is_2d = _get_coords_info(ds)
            
            for var in available_vars:
                try:
                    df = _extract_variable_data(ds, var, lons, lats, point_ids, lon_coord, lat_coord, is_2d)
                    all_data.append(df)
                except Exception as e:
                    print(f"  Warning: Could not extract {var}: {e}")
        
        if not all_data:
            raise ValueError(f"No data extracted from {grib_file}")
        
        # Combine all variables into long format
        df_long = pd.concat(all_data, ignore_index=True)
        
        # Process wind components (u/v -> speed/direction)
        df_long = _process_wind_components(df_long)
        
        # Reshape to wide format (one row per station-timestamp)
        df_wide = df_long.pivot_table(
            index=['ID', 'timestamp'], columns='variable', values='value', aggfunc='first'
        ).reset_index()
        
        # Convert temperature from Kelvin to Celsius if needed
        if 'TA' in df_wide.columns and df_wide['TA'].mean() > TEMP_KELVIN_THRESHOLD:
            df_wide['TA'] = df_wide['TA'] - KELVIN_TO_CELSIUS
        
        # Ensure all required SMET columns exist
        required_cols = ['ID', 'timestamp'] + REQUIRED_SMET_COLS
        df_wide = df_wide.reindex(columns=required_cols, fill_value=np.nan)
        
        print(f"  Done in {(datetime.now() - start_time).total_seconds():.1f} seconds")
        return df_wide[required_cols]
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# SMET Writing
# ============================================================================

def format_for_smet(df):
    """Format data for SMET output"""
    df = df.copy()
    # Round all data variables to 2 decimal places
    for col in REQUIRED_SMET_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    return df.fillna(MISSING_VALUE)


def _format_smet_value(val, col):
    """Format a single value for SMET output"""
    if pd.isna(val) or val == MISSING_VALUE:
        return str(MISSING_VALUE)
    # All values rounded to 2 decimal places
    if val == 0:
        return '0'
    return f"{val:.2f}".rstrip('0').rstrip('.')


def write_smet_file(filepath, station_id, latitude, longitude, altitude, data):
    """Write SMET file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("SMET 1.1 ASCII\n[HEADER]\n")
        f.write(f"station_id       = {station_id}\n")
        f.write(f"latitude         = {latitude}\n")
        f.write(f"longitude        = {longitude}\n")
        f.write(f"altitude         = {round(altitude)}\n")
        f.write(f"nodata           = {MISSING_VALUE}\n")
        f.write(f"fields = timestamp {' '.join(REQUIRED_SMET_COLS)}\n[DATA]\n")
        
        # Write data rows
        for _, row in data.iterrows():
            ts = row['timestamp'].strftime('%Y-%m-%dT%H:%M:%SZ')
            vals = [ts] + [_format_smet_value(row.get(col, MISSING_VALUE), col) for col in REQUIRED_SMET_COLS]
            f.write(' '.join(vals) + '\n')




# ============================================================================
# Main Entry Points
# ============================================================================

def _process_single_day(args):
    """Helper function for multiprocessing: extract one GRIB file"""
    grib_file, stations = args
    try:
        result = extract_grib_file(grib_file, stations)
        log_message(f"Processed {Path(grib_file).name}")
        return result
    except Exception as e:
        log_message(f"Error processing {Path(grib_file).name}: {e}", "ERROR")
        return None


def process_season(season, geojson_file, grib_dir, output_dir, n_procs=None):
    """Process entire season: extract all GRIB files and write final SMET files.
    
    Args:
        season: Season year (e.g., 2023)
        geojson_file: Path to GeoJSON file with station locations
        grib_dir: Directory containing HRDPS GRIB files
        output_dir: Output directory for SMET files
        n_procs: Number of parallel processes (default: auto-detect)
    
    Workflow:
        1. Load stations and extract elevations
        2. Process all GRIB files in parallel
        3. Combine data and create complete hourly time series
        4. Write SMET files (one per station)
    """
    import geopandas as gpd
    
    start_time = datetime.now()
    log_message(f"Starting season {season} processing")
    
    start_date, end_date = get_season_dates(season)
    log_message(f"Season {season}: {start_date.date()} to {end_date.date()}")
    
    stations = ensure_station_id(gpd.read_file(geojson_file))
    log_message(f"Loaded {len(stations)} stations")
    
    elevations = extract_station_elevations(get_dem_file(season), stations)
    
    # Get list of all GRIB files to process
    grib_dir_path = Path(grib_dir)
    grib_files = []
    for date in pd.date_range(start_date, end_date, freq='D'):
        grib_file = grib_dir_path / f"HRDPS_{date.strftime('%Y%m%d')}.grib2"
        if grib_file.exists():
            grib_files.append(grib_file)
        else:
            log_message(f"Missing: {grib_file.name}", "WARNING")
    
    if not grib_files:
        log_message("No GRIB files found", "ERROR")
        return
    
    # Process all days in parallel using multiprocessing
    if n_procs is None:
        n_procs = min(len(grib_files), int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count())))
    
    log_message(f"Processing {len(grib_files)} days with {n_procs} processes")
    
    with Pool(processes=n_procs) as pool:
        args_list = [(grib_file, stations) for grib_file in grib_files]
        results = pool.map(_process_single_day, args_list)
        
    all_data = [r for r in results if r is not None]
    
    if not all_data:
        log_message("No data extracted", "ERROR")
        return
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates(subset=['ID', 'timestamp'], keep='first')
    log_message(f"Combined {len(combined)} rows from {len(all_data)} files")
    
    # Create complete hourly time series
    hourly_timestamps = pd.date_range(start_date, end_date + timedelta(days=1), freq='H', inclusive='left')
    
    # Write final SMET files directly (no chunks)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    for station_id, sdata in combined.groupby('ID'):
        # Merge with complete time series to fill gaps
        complete_df = pd.DataFrame({'timestamp': hourly_timestamps})
        complete_df = complete_df.merge(sdata, on='timestamp', how='left')
        complete_df = complete_df.reindex(columns=['timestamp'] + REQUIRED_SMET_COLS, fill_value=pd.NA)
        complete_df[REQUIRED_SMET_COLS] = complete_df[REQUIRED_SMET_COLS].fillna(MISSING_VALUE)
        complete_df = format_for_smet(complete_df)
        
        # Get station coordinates and elevation
        station_row = stations[stations['station_id'] == station_id]
        if not station_row.empty:
            station_row = station_row.iloc[0]
            write_smet_file(output_dir_path / f"{station_id}.smet", station_id,
                           station_row.geometry.y, station_row.geometry.x,
                           elevations.get(station_id, 0), complete_df)
            log_message(f"  Wrote {len(complete_df)} timesteps for {station_id}")
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    log_message(f"Season {season} complete ({duration:.1f} min)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Archive2SMET pipeline - process entire season')
    parser.add_argument('--season', type=int, required=True, help='Season year (e.g., 2023)')
    parser.add_argument('--geojson', type=str, required=True, help='Path to GeoJSON file with stations')
    parser.add_argument('--grib-dir', type=str, required=True, help='Directory containing HRDPS GRIB files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for SMET files')
    
    args = parser.parse_args()
    
    # Use SLURM_CPUS_PER_TASK if available, otherwise auto-detect
    n_procs = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    process_season(args.season, args.geojson, args.grib_dir, args.output_dir, n_procs)

