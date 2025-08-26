# I have the following files in the output directory:
'''
2019_B01_WGS84.tif
2019_B02_WGS84.tif
2019_B03_WGS84.tif
2019_B04_WGS84.tif
2019_B05_WGS84.tif
2019_B06_WGS84.tif
2019_B07_WGS84.tif

2020_B01_WGS84.tif
2020_B02_WGS84.tif
...
2023_B07_WGS84.tif

Stack all the bands for each year and save the merged tiff file as netcdf in the output directory

'''

import os
import glob
import rasterio
import xarray as xr
import numpy as np
from rasterio.merge import merge

def stack_bands_and_save_netcdf(year, output_dir):
    # Find all TIFF files for the given year
    files = glob.glob(os.path.join(output_dir, f"{year}_B*_WGS84.tif"))
    files.sort()  # Ensure consistent order

    # Open all rasters
    src_files_to_mosaic = [rasterio.open(file) for file in files]

    # Merge rasters
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Close raster files
    for src in src_files_to_mosaic:
        src.close()

    # Create xarray DataArray
    bands = [f"Band{i}" for i in range(1, 8)]
    da = xr.DataArray(
        mosaic,
        dims=('band', 'y', 'x'),
        coords={
            'band': bands,
            'y': np.arange(mosaic.shape[1]),
            'x': np.arange(mosaic.shape[2])
        }
    )

    # Set attributes
    da.attrs['transform'] = out_trans
    da.attrs['crs'] = src_files_to_mosaic[0].crs.to_string()

    # Create Dataset
    ds = da.to_dataset(name="modis")

    # Save as NetCDF
    output_file = os.path.join(output_dir, f"{year}_stacked_modis.nc")
    ds.to_netcdf(output_file)
    print(f"Saved {output_file}")



# Specify the output directory
output_dir = "./final_modis_data"

# Process each year
for year in range(2019, 2024):
    stack_bands_and_save_netcdf(str(year), output_dir)


