from logging_helper import setup_logger
from utilities import reproject_to_wgs84, resample_to_new_resolution
from tqdm import tqdm
import time
import os
import glob
import rasterio
from rasterio.merge import merge


def dir_diff(a, b):
    a = set(a)
    b = set(b)
    
    _diff = a.difference(b)
    
    #Remove all directories that start with "2024" from the difference
    _diff = [x for x in _diff if not x.startswith("2024")]
    
    return _diff


if __name__ == "__main__":
    # Set up logger
    logger = setup_logger("./logs/tif_project_local_follow_up.log")

    # Base directory
    base_dir_pattern = "../gis-stac/IA_sentinel1/2020-*"
    base_dirs = glob.glob(base_dir_pattern)
    
    #Output directory
    output_dir = "./process_s1"
    
    # a = os.listdir("../gis-stac/IA_sentinel1")
    # b = os.listdir("./process_s1")
    # diff = dir_diff(a, b)
    
    # Bands to merge
    bands = ["vv", "vh"]
    
    # Go to each directory within the base directory and merge the bands. There are several bands
    
    start_time = time.time()
    
    for folder in tqdm(base_dirs, desc="Processing folders"):
    # for folder in diff:
        
        #Extract the folder name from the path
        _folder = folder.split("/")[-1]
        
        os.makedirs(f"{output_dir}/{_folder}", exist_ok=True)

        
        print(f"Processing folder: {folder}")
        
        # Get the list of files for each band
        
        for band in bands:
            band_files = glob.glob(os.path.join(folder, f"*_{band}.tif"))
            if not band_files:
                print(f"No files found for band: {band}")
                continue
            else:
                #print(f"Found {len(band_files)} files for band: {band}")
                logger.info(f"Found {len(band_files)} files for band: {band}")
                for file in band_files:
                    try:
                        with rasterio.open(file) as src:
                            
                            #Get the CRS
                            crs = src.crs
                            
                            # Resample to 125m
                            resample_to_new_resolution(file, f"{output_dir}/{folder}/R125_{os.path.basename(file)}", crs ,125)
                            
                            # Reproject to WGS84
                            reproject_to_wgs84(f"{output_dir}/{folder}/R125_{os.path.basename(file)}", f"{output_dir}/{folder}/WGS84_{os.path.basename(file)}", crs)
                    
                    except Exception as e:
                        logger.error(f"Error processing {file}: {e}")
                        continue