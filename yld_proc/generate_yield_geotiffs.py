import pandas as pd
import numpy as np
from math import pi, cos, ceil
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import geopandas as gpd
import os
import glob

def create_geotiff_from_group(df):
    groups = df.groupby('Layer_ID')
    for layer_id, group in groups:
        if group.empty:
            continue
        
        min_lon = group['x'].min()
        max_lon = group['x'].max()
        min_lat = group['y'].min()
        max_lat = group['y'].max()
        avg_lat = group['y'].mean()
        
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * cos(avg_lat * pi / 180)
        res_m = 5.0
        res_lat = res_m / m_per_deg_lat
        res_lon = res_m / m_per_deg_lon
        nrows = ceil((max_lat - min_lat) / res_lat)
        ncols = ceil((max_lon - min_lon) / res_lon)
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, ncols, nrows)
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(min_lon, max_lon, ncols),
            np.linspace(max_lat, min_lat, nrows)
        )
        points = group[['x', 'y']].values
        values = group['Yield'].values
        
        # Perform linear interpolation
        interpolated = griddata(points, values, (lon_grid, lat_grid), method='linear')
        
        # Create a mask based on distance from actual data points
        # Only keep interpolated values within a reasonable distance from real data
        grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        tree = cKDTree(points)
        
        # Calculate distances to nearest data point
        # Convert maximum distance threshold to degrees (approximately 3 pixels worth)
        max_distance_deg = 3 * max(res_lon, res_lat)
        distances, _ = tree.query(grid_points)
        distances = distances.reshape(lon_grid.shape)
        
        # Apply mask: only keep interpolated values close to actual data points
        # This prevents interpolation in areas with no data
        data = np.where(distances <= max_distance_deg, interpolated, np.nan)
        
        # Replace remaining NaN with 0 (for areas with valid interpolation that failed)
        data = np.nan_to_num(data, nan=0.0)
        os.makedirs('unprocessed_data/yield_geotiffs', exist_ok=True)
        if len(group.Crop.unique()) == 1:
            output_file = f"unprocessed_data/yield_geotiffs/{layer_id}_{group['Crop'].iloc[0]}.tif"
        else:
            output_file = f'unprocessed_data/yield_geotiffs/{layer_id}_mixed.tif'
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
        print(f'Created GeoTIFF: {output_file}')

# Process all available years
parquet_files = sorted(glob.glob('Yield_*_filtered.parquet'))
# parquet_files = sorted(glob.glob('Yield_2019_filtered.parquet'))

if not parquet_files:
    print("No filtered yield parquet files found.")
else:
    for parquet_file in parquet_files:
        year = parquet_file.split('_')[1]
        print(f"Processing year: {year}")
        full_data = gpd.read_parquet(parquet_file)
        create_geotiff_from_group(full_data)