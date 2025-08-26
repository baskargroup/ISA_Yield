import os
import requests
import rasterio
import glob
from rasterio.merge import merge
import time
from rasterio.warp import calculate_default_transform, reproject, Resampling
import sys
import matplotlib.pyplot as plt
from logging_helper import setup_logger
from tqdm import tqdm

def merge_bands(list_of_files, band_name, output_dir):
    # Merge images
    src_files = [rasterio.open(f) for f in list_of_files]
    mosaic, out_trans = merge(src_files, method="max")

    # Save merged image
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    write_path = os.path.join(output_dir, f"{band_name}.tif")
    
    with rasterio.open(write_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Merged image saved: {write_path}")

if __name__ == "__main__":
    # Set up logger
    logger = setup_logger("./logs/s2_merge_local.log")

    # Base directory
    base_dir = "./processed_s2"
    
    #Output directory
    output_dir = "./final_s2"
    
    # Bands to merge
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    
    # Go to each directory within the base directory and merge the bands. There are several bands
    
    start_time = time.time()
    
    for folder in tqdm(os.listdir(base_dir), desc="Processing folders"):
        
        os.makedirs(f"{output_dir}/{folder}", exist_ok=True)

        folder_path = os.path.join(base_dir, folder)
        print(f"Processing folder: {folder_path}")
        
        # Get the list of files for each band
        
        for band in bands:
            band_files = glob.glob(os.path.join(folder_path, f"WGS84_*_{band}.tif"))
            if not band_files:
                print(f"No files found for band: {band}")
                continue
            else:
                #print(f"Found {len(band_files)} files for band: {band}")
                logger.info(f"Found {len(band_files)} files for band: {band}")
                
                # Merge the bands
                try:
                    output_path = os.path.join(output_dir, folder)
                    merge_bands(band_files, band, output_path)
                    logger.info(f"Merged bands for band: {band} in folder: {output_path}")
                
                except Exception as e:
                    print(f"Error merging bands: {e}")
                    logger.error(f"Error merging bands: {e}")
                    continue
         
    print(f"Time taken: {time.time() - start_time} seconds")