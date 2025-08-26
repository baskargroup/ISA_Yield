import os
import rasterio
import glob
from rasterio.merge import merge
import time
from rasterio.warp import calculate_default_transform, reproject, Resampling
import sys
import matplotlib.pyplot as plt
from logging_helper import setup_logger
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    start = time.time()
    
    define_bands = {
        "sentinel2": ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
        "sentinel1": ["vv","vh"],
        "modis_LAI": ["Lai_500m","Fpar_500m"],
        "modis_SURF": ["sur_refl_b01","sur_refl_b02"],
        "landsat8": ["B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12"],
        "modis_SURF": ["sur_refl_b01","sur_refl_b02"],
        "modis_LANDSURF": ["sur_refl_b01","sur_refl_b02"],
        "modis_NBAR": ['Band1','Band2','Band3','Band4','Band5','Band6','Band7'], 
    }
    
    base_dir = "/work/mech-ai-scratch/rtali/gis-stac/IA_modis_NBAR"
    product_name = "modis_NBAR"
    output_dir = f"./processed_modis"
    
    # Set up the logging
    my_logger = setup_logger(f"./logs/modis.log")
    
    # We now have the reprojected tiff files in the output directory for every week
    # We will now merge the tiff files for each year by creating a dimension of time
    # We will then save the merged tiff files in the output directory
    year = [2019, 2020, 2021, 2022, 2023]
    
    # For a given year choose the directories that have the year in their name
    for _year in year:
        dirs = [d for d in os.listdir(output_dir) if str(_year) in d]
        print(f"Total number of directories for year {_year}: {len(dirs)}")
        
        for _band in define_bands[product_name]:
            list_of_files = []
            for d in tqdm(dirs):
                # Get the list of tiff files for each band
                tiff_files = glob.glob(f"{output_dir}/{d}/{_band}_WGS84.tif")   
           
                if len(tiff_files) == 0:
                    print(f"No tiff files found for directory: {d}")
                    my_logger.info(f"No tiff files found for directory: {d}")
                    continue
                
                list_of_files.append(tiff_files[0])
                
            # Merge the tiff files
            merged_tiff_file = f"{output_dir}/{_year}_{_band}.tif"
            
            try:
                # Open each file to get their data and create a list of bands
                src_files_to_mosaic = [rasterio.open(f) for f in list_of_files]
                
                # Prepare an array to hold the bands (raster data)
                bands = []
                for src in src_files_to_mosaic:
                    bands.append(src.read(1))  # Read the first band of each file
                
                # Stack the bands into one 3D array (number of bands, height, width)
                mosaic = np.stack(bands)
                
                #print the shape of the mosaic
                print(f"Shape of the mosaic for year {_year}: {mosaic.shape}")
                
                # Copy the metadata from the first file
                out_meta = src_files_to_mosaic[0].meta.copy()
                
                # Update metadata to reflect the number of bands
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "count": mosaic.shape[0],  # Number of bands (files)
                    "transform": src_files_to_mosaic[0].transform,
                })
                
                # Write the mosaic raster to disk with the new bands
                with rasterio.open(merged_tiff_file, "w", **out_meta) as dest:
                    for i in range(mosaic.shape[0]):
                        dest.write(mosaic[i], i + 1)  # Write each band separately
                
                print(f"Successfully merged tiff files for year {_year} and band {_band}")
                my_logger.info(f"Successfully merged tiff files for year {_year} and band {_band}")
                
            except Exception as e:
                my_logger.error(f"Failed to merge tiff files for year {_year} and band {_band}. Error: {e}")
                continue
            
            finally:
                # Close all opened raster files
                for src in src_files_to_mosaic:
                    src.close()
