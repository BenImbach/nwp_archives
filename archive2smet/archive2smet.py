#!/usr/bin/env python3
"""
Core functions for archive2smet pipeline
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import tempfile
import shutil
import warnings

warnings.filterwarnings('ignore')

# Constants
KELVIN_TO_CELSIUS = 273.15
TEMP_KELVIN_THRESHOLD = 100  # If mean temp > 100, assume Kelvin
MISSING_VALUE = -999
REQUIRED_SMET_COLS = ['DW', 'ILWR', 'ISWR', 'PSUM', 'RH', 'TA', 'VW']


# ============================================================================
# Date/Season Utilities
# ============================================================================

def get_season_dates(season):
    """Get start (Sept 1 prev year) and end (May 31 season year) dates"""
    return datetime(int(season) - 1, 9, 1), datetime(int(season), 5, 31)


def get_chunk_dates(season, chunk_id, total_chunks):
    """Get date range for chunk"""
    start_date, end_date = get_season_dates(season)
    total_days = (end_date - start_date).days + 1
    days_per_chunk = total_days / total_chunks
    chunk_start = (start_date + timedelta(days=int((chunk_id - 1) * days_per_chunk))).date()
    chunk_end = (start_date + timedelta(days=int(chunk_id * days_per_chunk) - 1)).date()
    return chunk_start, chunk_end


def calculate_total_chunks(season, days_per_chunk=7):
    """Calculate total chunks for season"""
    start_date, end_date = get_season_dates(season)
    total_days = (end_date - start_date).days + 1
    return int((total_days + days_per_chunk - 1) / days_per_chunk)


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
        if 'id' in stations.columns:
            stations['station_id'] = stations['id']
        else:
            stations['station_id'] = [f"station_{i}" for i in range(len(stations))]
    return stations


def log_message(log_file, message, level="INFO"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | {level} | {message}"
    print(log_entry)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")


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
    
    # Get coordinates
    lons, lats = stations.geometry.x.values, stations.geometry.y.values
    if 'station_id' in stations.columns:
        station_ids = stations['station_id'].values
    else:
        station_ids = np.array([f"station_{i}" for i in range(len(stations))])
    lon_coord = 'longitude' if 'longitude' in ds.coords else 'x'
    lat_coord = 'latitude' if 'latitude' in ds.coords else 'y'
    
    # Extract (handle 2D vs 1D coordinates)
    lon_dims, lat_dims = ds.coords[lon_coord].dims, ds.coords[lat_coord].dims
    is_2d = len(lon_dims) == 2 and len(lat_dims) == 2
    
    if is_2d:
        grid_coords = np.column_stack([ds.coords[lon_coord].values.flatten(),
                                      ds.coords[lat_coord].values.flatten()])
        _, indices = cKDTree(grid_coords).query(np.column_stack([lons, lats]))
        y_size, x_size = ds.coords[lon_coord].shape
        elev_data = ds[elev_var].isel(y=xr.DataArray(indices // x_size, dims='points'),
                                      x=xr.DataArray(indices % x_size, dims='points'))
    else:
        elev_data = ds[elev_var].sel({lon_coord: xr.DataArray(lons, dims='points'),
                                     lat_coord: xr.DataArray(lats, dims='points')}, method='nearest')
    
    elevations = elev_data.values
    print(f"  Extracted elevations: {min(elevations):.1f} - {max(elevations):.1f} m")
    return {sid: float(elev) for sid, elev in zip(station_ids, elevations)}


# ============================================================================
# GRIB Extraction
# ============================================================================

def extract_grib_file(grib_file, stations, point_id_col="station_id"):
    """Extract GRIB data at station locations"""
    print(f"Extracting {len(stations)} points from {grib_file}")
    start_time = datetime.now()
    
    variable_mapping = {
        't': 'TA', 't2m': 'TA', 'r': 'RH', 'r2': 'RH', '2r': 'RH',
        'ws': 'VW', 'wdir': 'DW', 'wdir10': 'DW',
        'u10': 'VW_U', 'v10': 'VW_V', 'si10': 'VW_ALT',
        'sdswrf': 'ISWR', 'sdlwrf': 'ILWR', 'dswrf': 'ISWR', 'dlwrf': 'ILWR',
        'unknown': 'PSUM', 'tp': 'PSUM', 'prate': 'PSUM',
    }
    
    temp_dir = tempfile.mkdtemp()
    try:
        index_path = os.path.join(temp_dir, Path(grib_file).name + '.idx')
        
        # Open GRIB datasets
        datasets = []
        try:
            datasets.append(xr.open_dataset(grib_file, engine='cfgrib', 
                                           backend_kwargs={'indexpath': index_path}))
        except:
            for level in ['surface', 'heightAboveGround', 'atmosphere']:
                try:
                    level_path = os.path.join(temp_dir, f"{Path(grib_file).stem}.{level}.idx")
                    datasets.append(xr.open_dataset(grib_file, engine='cfgrib',
                                                   backend_kwargs={'filter_by_keys': {'typeOfLevel': level},
                                                                  'indexpath': level_path}))
                except:
                    pass
        
        if not datasets:
            raise ValueError(f"Could not open {grib_file}")
        
        # Extract coordinates
        lons, lats = stations.geometry.x.values, stations.geometry.y.values
        point_ids = stations[point_id_col].values
        
        # Extract data
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
                _, indices = cKDTree(grid_coords).query(np.column_stack([lons, lats]))
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
                        var_data = ds[var].sel({lon_coord: xr.DataArray(lons, dims='points'),
                                               lat_coord: xr.DataArray(lats, dims='points')}, method='nearest')
                    
                    df = var_data.to_dataframe(name='value').reset_index()
                    df['variable'] = variable_mapping[var]
                    if 'points' in df.columns:
                        df['ID'] = df['points'].map(lambda idx: point_ids[idx])
                    else:
                        # Replicate point_ids for each timestep
                        n_timesteps = len(df) // len(point_ids)
                        df['ID'] = np.tile(point_ids, n_timesteps)[:len(df)]
                    df['timestamp'] = pd.to_datetime(df[time_coord])
                    all_data.append(df[['ID', 'timestamp', 'variable', 'value']])
                except Exception as e:
                    print(f"  Warning: Could not extract {var}: {e}")
        
        if not all_data:
            raise ValueError(f"No data extracted from {grib_file}")
        
        # Combine and reshape
        df_long = pd.concat(all_data, ignore_index=True)
        
        # Handle wind components
        if 'VW_U' in df_long['variable'].values and 'VW_V' in df_long['variable'].values:
            wind_df = df_long[df_long['variable'].isin(['VW_U', 'VW_V'])].pivot_table(
                index=['ID', 'timestamp'], columns='variable', values='value', aggfunc='first').reset_index()
            wind_df['VW'] = np.sqrt(wind_df['VW_U']**2 + wind_df['VW_V']**2)
            wind_df['DW'] = (270 - np.arctan2(wind_df['VW_V'], wind_df['VW_U']) * 180 / np.pi) % 360
            wind_long = wind_df[['ID', 'timestamp', 'VW', 'DW']].melt(
                id_vars=['ID', 'timestamp'], var_name='variable', value_name='value')
            df_long = pd.concat([df_long[~df_long['variable'].isin(['VW_U', 'VW_V'])], wind_long], ignore_index=True)
        
        # Reshape to wide format
        df_wide = df_long.pivot_table(index=['ID', 'timestamp'], columns='variable', values='value', 
                                     aggfunc='first').reset_index()
        
        # Convert temperature from Kelvin if needed
        if 'TA' in df_wide.columns and df_wide['TA'].mean() > TEMP_KELVIN_THRESHOLD:
            df_wide['TA'] = df_wide['TA'] - KELVIN_TO_CELSIUS
        
        # Ensure all required columns
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
    for col in ['DW', 'ILWR', 'ISWR', 'RH']:
        if col in df.columns:
            df[col] = np.floor(pd.to_numeric(df[col], errors='coerce')).astype('Int64')
    for col in ['PSUM', 'TA', 'VW']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2 if col == 'PSUM' else 1)
    return df.fillna(MISSING_VALUE)


def write_smet_file(filepath, station_id, latitude, longitude, altitude, data):
    """Write SMET file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("SMET 1.1 ASCII\n[HEADER]\n")
        f.write(f"station_id       = {station_id}\n")
        f.write(f"latitude         = {latitude:.6g}\n")
        f.write(f"longitude        = {longitude:.6g}\n")
        f.write(f"altitude         = {int(altitude)}\n")
        f.write(f"nodata           = {MISSING_VALUE}\n")
        f.write(f"fields = timestamp {' '.join(REQUIRED_SMET_COLS)}\n[DATA]\n")
        
        # Write data
        data = data.copy()
        data['ts'] = data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        for _, row in data.iterrows():
            vals = [row['ts']]
            for col in REQUIRED_SMET_COLS:
                val = row.get(col, MISSING_VALUE)
                if pd.isna(val) or val == MISSING_VALUE:
                    vals.append(str(MISSING_VALUE))
                elif col == 'PSUM':
                    vals.append('0' if val == 0 else f"{val:.2f}".rstrip('0').rstrip('.'))
                elif col in ['TA', 'VW']:
                    vals.append(f"{val:.1f}")
                else:
                    vals.append(f"{int(val)}")
            f.write(' '.join(vals) + '\n')


def read_smet_data(filepath):
    """Read data section from SMET file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_start = next((i + 1 for i, line in enumerate(lines) if line.strip() == '[DATA]'), None)
    if data_start is None:
        return pd.DataFrame()
    
    data_rows = []
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        try:
            data_rows.append({
                'timestamp': pd.to_datetime(parts[0]),
                'DW': float(parts[1]) if parts[1] != str(MISSING_VALUE) else pd.NA,
                'ILWR': float(parts[2]) if parts[2] != str(MISSING_VALUE) else pd.NA,
                'ISWR': float(parts[3]) if parts[3] != str(MISSING_VALUE) else pd.NA,
                'PSUM': float(parts[4]) if parts[4] != str(MISSING_VALUE) else pd.NA,
                'RH': float(parts[5]) if parts[5] != str(MISSING_VALUE) else pd.NA,
                'TA': float(parts[6]) if parts[6] != str(MISSING_VALUE) else pd.NA,
                'VW': float(parts[7]) if parts[7] != str(MISSING_VALUE) else pd.NA,
            })
        except:
            continue
    
    return pd.DataFrame(data_rows)


# ============================================================================
# Main Entry Points
# ============================================================================

def process_chunk(season, chunk_id, geojson_file, grib_dir, output_dir, log_file):
    """Process one time chunk: extract GRIB data and write chunk SMET files"""
    import geopandas as gpd
    
    start_time = datetime.now()
    log_message(log_file, f"Starting chunk {chunk_id} processing")
    
    total_chunks = calculate_total_chunks(season)
    chunk_start, chunk_end = get_chunk_dates(season, chunk_id, total_chunks)
    log_message(log_file, f"Chunk {chunk_id}/{total_chunks}: {chunk_start} to {chunk_end}")
    
    stations = ensure_station_id(gpd.read_file(geojson_file))
    log_message(log_file, f"Loaded {len(stations)} stations")
    
    elevations = extract_station_elevations(get_dem_file(season), stations)
    
    all_data = []
    grib_dir_path = Path(grib_dir)
    for date in pd.date_range(chunk_start, chunk_end, freq='D'):
        grib_file = grib_dir_path / f"HRDPS_{date.strftime('%Y%m%d')}.grib2"
        if grib_file.exists():
            try:
                all_data.append(extract_grib_file(grib_file, stations))
                log_message(log_file, f"Processed {grib_file.name}")
            except Exception as e:
                log_message(log_file, f"Error processing {grib_file.name}: {e}", "ERROR")
        else:
            log_message(log_file, f"Missing: {grib_file.name}", "WARNING")
    
    if not all_data:
        log_message(log_file, "No data extracted", "ERROR")
        return
    
    combined = format_for_smet(pd.concat(all_data, ignore_index=True))
    log_message(log_file, f"Combined {len(combined)} rows from {len(all_data)} files")
    
    chunk_dir = Path(output_dir) / "chunks" / f"chunk_{chunk_id:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    for station_id, sdata in combined.groupby('ID'):
        station_row = stations[stations['station_id'] == station_id]
        if len(station_row) > 0:
            station_row = station_row.iloc[0]
            write_smet_file(chunk_dir / f"{station_id}.smet", station_id,
                           station_row.geometry.y, station_row.geometry.x,
                           elevations.get(station_id, 0),
                           sdata.drop(columns=['ID']))
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    log_message(log_file, f"Chunk {chunk_id} complete ({duration:.1f} min)")


def concatenate_chunks(chunk_dir, stations, season, output_dir, log_file):
    """Merge chunk SMET files into final station files"""
    import geopandas as gpd
    
    start_time = datetime.now()
    log_message(log_file, "Starting concatenation")
    
    elevations = extract_station_elevations(get_dem_file(season), stations)
    
    chunk_dirs = sorted([d for d in Path(chunk_dir).iterdir() 
                         if d.is_dir() and d.name.startswith('chunk_')])
    log_message(log_file, f"Found {len(chunk_dirs)} chunk directories")
    
    all_station_ids = {f.stem for chunk_path in chunk_dirs 
                      for f in chunk_path.glob("*.smet")}
    log_message(log_file, f"Processing {len(all_station_ids)} stations")
    
    start_date, end_date = get_season_dates(season)
    hourly_timestamps = pd.date_range(pd.Timestamp(start_date), 
                                      pd.Timestamp(end_date) + pd.Timedelta(days=1),
                                      freq='H', inclusive='left')
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    for station_id in sorted(all_station_ids):
        all_data = []
        for chunk_path in chunk_dirs:
            chunk_file = chunk_path / f"{station_id}.smet"
            if chunk_file.exists():
                try:
                    df = read_smet_data(chunk_file)
                    if len(df) > 0:
                        all_data.append(df)
                except Exception as e:
                    log_message(log_file, f"Error reading {chunk_file}: {e}", "WARNING")
        
        if not all_data:
            log_message(log_file, f"No data for {station_id}", "WARNING")
            continue
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
        
        complete_df = pd.DataFrame({'timestamp': hourly_timestamps})
        complete_df = complete_df.merge(combined, on='timestamp', how='left')
        
        complete_df = complete_df.reindex(columns=['timestamp'] + REQUIRED_SMET_COLS, fill_value=pd.NA)
        complete_df[REQUIRED_SMET_COLS] = complete_df[REQUIRED_SMET_COLS].fillna(MISSING_VALUE)
        
        complete_df = format_for_smet(complete_df)
        
        # Get station coordinates and elevation from DEM
        station_row = stations[stations['station_id'] == station_id]
        if len(station_row) > 0:
            station_row = station_row.iloc[0]
            write_smet_file(output_dir_path / f"{station_id}.smet", station_id,
                           station_row.geometry.y, station_row.geometry.x,
                           elevations.get(station_id, 0), complete_df)
            log_message(log_file, f"  Wrote {len(complete_df)} timesteps for {station_id}")
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    log_message(log_file, f"Concatenation complete ({duration:.1f} min)")


if __name__ == "__main__":
    import argparse
    import geopandas as gpd
    
    parser = argparse.ArgumentParser(description='Archive2SMET pipeline')
    parser.add_argument('mode', choices=['extract', 'concat'], help='extract or concat')
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--geojson', type=str, required=True)
    parser.add_argument('--grib-dir', type=str)
    parser.add_argument('--chunk-dir', type=str)
    parser.add_argument('--chunk-id', type=int)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--log-file', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'extract':
        if not args.chunk_id or not args.grib_dir:
            parser.error("extract mode requires --chunk-id and --grib-dir")
    
    if not args.log_file:
        # Default to separate logs directory (not in output_dir), organized by season
        # Shell scripts should pass --log-file explicitly
        log_base = Path(os.environ.get('ARCHIVE2SMET_LOG_DIR', 
                                        Path.home() / 'scratch' / 'archive2smet' / 'logs' / str(args.season)))
        log_base.mkdir(parents=True, exist_ok=True)
        if args.mode == 'extract':
            args.log_file = log_base / f"chunk_{args.chunk_id:03d}.log"
        else:
            args.log_file = log_base / "concatenate.log"
    
    if args.mode == 'extract':
        process_chunk(args.season, args.chunk_id, args.geojson, args.grib_dir, args.output_dir, args.log_file)
    else:
        if not args.chunk_dir:
            parser.error("concat mode requires --chunk-dir")
        stations = ensure_station_id(gpd.read_file(args.geojson))
        concatenate_chunks(args.chunk_dir, stations, args.season, args.output_dir, args.log_file)

