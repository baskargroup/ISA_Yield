import glob
import time
import os
import rasterio
from rasterio.merge import merge
from tqdm import tqdm
from logging_helper import setup_logger
# from utilities import reproject_to_wgs84, resample_to_new_resolution
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
import time
import xarray as xr
import rioxarray as rxr
import os
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
import pandas as pd


def merge_files(list_of_files, crs, band_name, output_dir):
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

    write_path = os.path.join(output_dir, f"{crs}_{band_name}.tif")

    with rasterio.open(write_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Merged image saved: {write_path}")


if __name__ == "__main__":

    # Set up logger
    logger = setup_logger("./logs/modis_merge_local.log")

    # Base directory
    base_dir_pattern = "./reprojected_modis/2023-*"
    base_dirs = glob.glob(base_dir_pattern)

    # Sort the directories
    base_dirs.sort()

    # Output directory
    output_dir = "./modis"

    # a = os.listdir("../gis-stac/IA_sentinel1")
    # b = os.listdir("./process_s1")
    # diff = dir_diff(a, b)

    # Bands to merge
    bands = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7']

    # Go to each directory within the base directory and merge the bands. There are several bands

    start_time = time.time()

    print(f"Start Time in CDT: {time.ctime(start_time)}")

    for folder in tqdm(base_dirs, desc="Processing folders"):
        # for folder in diff:

        # Extract the folder name from the path
        _folder = folder.split("/")[-1]

        os.makedirs(f"{output_dir}/{_folder}", exist_ok=True)

        print(f"Processing folder: {folder}")

        # Get the list of files for each band

        for band in bands:

            band_processing_time = time.time()

            # Get the list of files for each band
            band_files = glob.glob(os.path.join(folder, f"*_{band}.tif"))

            if not band_files:
                print(f"No files found for band: {band}")
                logger.info(f"No files found for band: {band}")
                continue

            else:

                files_to_merge = []
                grand_merge_time = time.time()

                for file in band_files:
                    files_to_merge.append(file)

                # print(f"Files to merge: {files_to_merge}")
                os.makedirs(f"./modis/{_folder}", exist_ok=True)
                merge_files(files_to_merge, "4326", band, f"./modis/{_folder}")

                print(
                    f"Grand merge complete! Time taken: {time.time() - grand_merge_time} seconds")
                print("\n")

            print(
                f"Processing complete for band: {band}. Time taken: {time.time() - band_processing_time} seconds")

    end_time = time.time()
    print(
        f"Processing complete! Total time taken: {end_time - start_time} seconds")
    logger.info(
        f"Processing complete! Total time taken: {end_time - start_time} seconds")
