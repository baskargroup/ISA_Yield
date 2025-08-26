import glob
import time
import os
import rasterio
from rasterio.merge import merge
from tqdm import tqdm
from logging_helper import setup_logger
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
from datetime import datetime
import numpy as np
import pdb

def check_date(file_name, dt):
    _date = file_name.split("_")[1]
    _date = _date.split("T")[0]
    print(f"Date: {_date}")
    _date = pd.to_datetime(_date, format="%Y-%m-%d")
    dt = pd.to_datetime(str(dt))
    if np.abs((_date - dt).days) <= 7:
        return True
    return False

def reproject_to_wgs84(input_path, output_path, src_crs, dst_crs="EPSG:4326"):
    src = rasterio.open(input_path)
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
    print(f"Conversion complete! Data reprojected to WGS84 and saved as {output_path}")

def merge_files(list_of_files, crs, band_name, output_dir, file_id=None):
    src_files = [rasterio.open(f) for f in list_of_files]
    mosaic, out_trans = merge(src_files, method="max")
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    if file_id:
        write_path = os.path.join(output_dir, f"{crs}_{band_name}_{file_id}.tif")
    else:
        write_path = os.path.join(output_dir, f"{crs}_{band_name}.tif")
    with rasterio.open(write_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    print(f"Merged image saved: {write_path}")
    return write_path

if __name__ == "__main__":
    logger = setup_logger("./logs/s1_merge_local_run2.log")
    base_dir_pattern = "./cropped/*"
    base_dirs = glob.glob(base_dir_pattern)
    base_dirs.sort()
    start = "2016-04-03"
    base_dirs = [dir for dir in base_dirs if dir.split("/")[-1] >= start]
    output_dir = "./process_s1"
    bands = ["vv", "vh"]
    
    start_time = time.time()
    print(f"Start Time in CDT: {time.ctime(start_time)}")
    
    for folder in tqdm(base_dirs, desc="Processing folders"):
        _folder = folder.split("/")[-1]
        os.makedirs(f"{output_dir}/{_folder}", exist_ok=True)
        print(f"Processing folder: {folder}")
        
        for band in bands:
            band_processing_time = time.time()
            band_files = glob.glob(os.path.join(folder, f"{band}_*.tif"))
            band_files = [file for file in band_files if check_date(file, _folder)]
            
            if not band_files:
                print(f"No files found for band: {band}")
                logger.info(f"No files found for band: {band}")
                continue
            
            # Group files by ID
            id_groups = {}
            for file in band_files:
                file_id = os.path.basename(file).split('_')[-1].split('.')[0]
                if file_id not in id_groups:
                    id_groups[file_id] = {'32614': [], '32615': [], '32616': []}
                
                try:
                    with rasterio.open(file) as src:
                        crs = src.crs.to_string()
                        if crs in ["EPSG:32614", "EPSG:32615", "EPSG:32616"]:
                            id_groups[file_id][crs.split(':')[1]].append(file)
                        else:
                            print(f"CRS {crs} not supported for file {file}")
                            continue
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    logger.error(f"Error processing file {file}: {e}")
                    continue
            
            # Process each ID group
            for file_id, utm_dict in id_groups.items():
                try:
                    merge_time = time.time()
                    merged_files = []
                    
                    for utm, files in utm_dict.items():
                        if files:
                            merged_path = merge_files(files, utm, band, f"{output_dir}/{_folder}", file_id)
                            merged_files.append(merged_path)
                    
                    # Reproject merged files to WGS84
                    reproject_time = time.time()
                    wgs84_files = []
                    for merged_file in merged_files:
                        crs = os.path.basename(merged_file).split('_')[0]
                        wgs84_path = f"{output_dir}/{_folder}/{crs}_WGS84_{band}_{file_id}.tif"
                        reproject_to_wgs84(merged_file, wgs84_path, f"EPSG:{crs}")
                        wgs84_files.append(wgs84_path)
                    
                    # Final merge of WGS84 files
                    os.makedirs(f"./final_s1/{_folder}", exist_ok=True)
                    if wgs84_files:
                        grand_merge_time = time.time()
                        merge_files(wgs84_files, "4326", band, f"./final_s1/{_folder}", file_id)
                        print(f"Grand merge complete for ID {file_id}! Time taken: {time.time() - grand_merge_time} seconds")
                    
                    # Remove intermediate files
                    remove_time = time.time()
                    for merged_file in merged_files:
                        if os.path.exists(merged_file):
                            os.remove(merged_file)
                    for wgs84_file in wgs84_files:
                        if os.path.exists(wgs84_file):
                            os.remove(wgs84_file)
                    print(f"Removed intermediate files for band {band}, ID {file_id}. Time taken: {time.time() - remove_time} seconds")
                    
                    print(f"Merged bands complete for ID {file_id}! Time taken: {time.time() - merge_time} seconds")
                except Exception as e:
                    print(f"Error processing ID {file_id}: {e}")
                    logger.error(f"Error processing ID {file_id}: {e}")
                    continue
            
            print(f"Processing complete for band: {band}. Time taken: {time.time() - band_processing_time} seconds")
    
    end_time = time.time()
    print(f"Processing complete! Total time taken: {end_time - start_time} seconds")
    logger.info(f"Processing complete! Total time taken: {end_time - start_time} seconds")