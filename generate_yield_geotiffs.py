import pandas as pd
import numpy as np
from math import pi, cos, ceil
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
import geopandas as gpd
import os
import pdb
def create_geotiff_from_group(df):
    groups = df.groupby('Layer_ID')
    # pdb.set_trace()
    for layer_id, group in groups:
        if group.empty:
            continue
        
        # Calculate bounds and average latitude
        min_lon = group['x'].min()
        max_lon = group['x'].max()
        min_lat = group['y'].min()
        max_lat = group['y'].max()
        avg_lat = group['y'].mean()
        
        # Approximate meters per degree (WGS84 equatorial values)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * cos(avg_lat * pi / 180)
        
        # Desired resolution in meters
        res_m = 60.0
        
        # Resolution in degrees
        res_lat = res_m / m_per_deg_lat
        res_lon = res_m / m_per_deg_lon
        
        # Calculate grid dimensions (add buffer to cover bounds)
        nrows = ceil((max_lat - min_lat) / res_lat) + 1
        ncols = ceil((max_lon - min_lon) / res_lon) + 1
        
        # Create affine transform (top-left origin)
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, ncols, nrows)
        
        # Create grid coordinates for interpolation (lats decrease from north to south)
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(min_lon, max_lon, ncols),
            np.linspace(max_lat, min_lat, nrows)
        )
        
        # Extract points and values
        points = group[['x', 'y']].values
        values = group['Yld_Vol_Dr'].values
        
        # Interpolate to grid (use 'nearest', 'linear', or 'cubic'; fill NaNs with 0)
        interpolated = griddata(points, values, (lon_grid, lat_grid), method='linear')
        data = np.nan_to_num(interpolated, nan=0.0)
        os.makedirs('final_data/yield_geotiffs', exist_ok= True)
        if len(group.Crop.unique()) == 1:
            # Write to GeoTIFF
            output_file = f"final_data/yield_geotiffs/{layer_id}_{group['Crop'].iloc[0]}.tif"
        else:
            # Write to GeoTIFF
            output_file = f'final_data/yield_geotiffs/{layer_id}_mixed.tif'
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

# Call the function with your DataFrame
# create_geotiff_from_group(df)

full_data = gpd.read_parquet('Yield_2014_filtered.parquet')

create_geotiff_from_group(full_data)