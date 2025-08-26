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


def check_date(file_name, dt):
    _date = file_name.split("_")[-6]
    _date = _date.split("T")[0]

    print(f"Date: {_date}")

    # Convert to datetime. _date is a string in the format YYYYMMDD
    _date = pd.to_datetime(_date, format="%Y%m%d")

    # Convert dt to datetime. dt is in the format YYYY-MM-DD
    dt = pd.to_datetime(str(dt))

    # Check if the _date is within 7 days of the dt
    if np.abs((_date - dt).days) <= 7:
        return True
    else:
        return False


def reproject_to_wgs84(input_path, output_path, src_crs, dst_crs="EPSG:4326"):
    """
    Reprojects a given TIFF file to a WGS84.

    Args:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path to save the reprojected TIFF file.
        src_crs (str): Source CRS.
        dst_crs (str): Destination CRS (default: "EPSG:4326").

    """

    src = rasterio.open(input_path)

    # Calculate transformation parameters
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, src.width, src.height, *src.bounds
    )

    # Define output metadata
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Write the reprojected file
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):  # Loop through bands
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

    print(
        f"Conversion complete! Data reprojected to WGS84 and saved as {output_path}")


def resample_to_new_resolution(input_tif, output_tif, src_crs, target_resolution=125):
    with rasterio.open(input_tif) as src:
        # Define the target CRS. Since we are resampling, the target CRS is the same as the source CRS
        dst_crs = src_crs

        # Compute the transform, width, and height for the new resolution
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *
            src.bounds, resolution=target_resolution
        )

        # print(f"Original resolution: {src.res}")
        # print(f"New resolution: {target_resolution}m")
        # print(f"New dimensions: {width}x{height}")

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        # Open the output file and reproject
        with rasterio.open(output_tif, "w", **out_meta) as dst:
            for i in range(1, src.count + 1):  # Loop through all bands
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear  # Change to nearest if categorical data
                )

    print(f"Reprojected and resampled image saved: {output_tif}")


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
    logger = setup_logger("./logs/IL_s1_merge_local.log")

    # Base directory
    base_dir_pattern = "/work/mech-ai-scratch/rtali/gis-sentinel1-IL/IL_sentinel1/*"
    # base_dir_pattern = "./IA_sentinel1/*"
    base_dirs = glob.glob(base_dir_pattern)

    # Sort the directories
    base_dirs.sort()

    # Specify processing start date
    start = "2019-04-07"

    # Remove directories that are less than the start date
    base_dirs = [dir for dir in base_dirs if dir.split("/")[-1] >= start]

    # Output directory
    os.makedirs("./process_s1_IL", exist_ok=True)
    output_dir = "./process_s1_IL"

    # a = os.listdir("../gis-stac/IA_sentinel1")
    # b = os.listdir("./process_s1")
    # diff = dir_diff(a, b)

    # Bands to merge
    bands = ["vv", "vh"]

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

            """
            Maintain 2 list per UTM
            """

            utm_32614 = []
            utm_32615 = []
            utm_32616 = []

            # Get the list of files for each band
            band_files = glob.glob(os.path.join(folder, f"*_{band}.tif"))

            # Check that dates from the file names are between _folder and _folder + 7 days
            # If not, remove the file from the list
            band_files = [
                file for file in band_files if check_date(file, _folder)]

            if not band_files:
                print(f"No files found for band: {band}")
                logger.info(f"No files found for band: {band}")
                continue

            else:
                for file in band_files:
                    try:
                        print(f"Opening {file}")
                        with rasterio.open(file) as src:

                            resample_time = time.time()

                            if src.crs.to_string() == "EPSG:32614":

                                # Resample to 125m
                                resample_to_new_resolution(
                                    file, f"{output_dir}/{_folder}/{src.crs}_R125_{os.path.basename(file)}", src.crs, 125)
                                utm_32614.append(
                                    f"{output_dir}/{_folder}/{src.crs}_R125_{os.path.basename(file)}")

                            elif src.crs.to_string() == "EPSG:32615":

                                # Resample to 125m
                                resample_to_new_resolution(
                                    file, f"{output_dir}/{_folder}/{src.crs}_R125_{os.path.basename(file)}", src.crs, 125)
                                utm_32615.append(
                                    f"{output_dir}/{_folder}/{src.crs}_R125_{os.path.basename(file)}")

                            elif src.crs.to_string() == "EPSG:32616":

                                # Resample to 125m
                                resample_to_new_resolution(
                                    file, f"{output_dir}/{_folder}/{src.crs}_R125_{os.path.basename(file)}", src.crs, 125)
                                utm_32616.append(
                                    f"{output_dir}/{_folder}/{src.crs}_R125_{os.path.basename(file)}")

                            else:
                                print("CRS not supported")
                                continue

                            print(
                                f"Resampling complete! Time taken: {time.time() - resample_time} seconds")

                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        logger.error(f"Error processing {file}: {e}")
                        continue

                # Merge the bands for each UTM
                try:

                    merge_time = time.time()

                    if len(utm_32614) > 0:
                        merge_files(utm_32614, "32614", band,
                                    f"{output_dir}/{_folder}")
                    if len(utm_32615) > 0:
                        merge_files(utm_32615, "32615", band,
                                    f"{output_dir}/{_folder}")
                    if len(utm_32616) > 0:
                        merge_files(utm_32616, "32616", band,
                                    f"{output_dir}/{_folder}")

                    # merge_files(utm_32614, "32614", band, f"{output_dir}/{_folder}")
                    # merge_files(utm_32615, "32615", band, f"{output_dir}/{_folder}")
                    # merge_files(utm_32616, "32616", band, f"{output_dir}/{_folder}")

                    print(
                        f"Merged bands complete! Time taken: {time.time() - merge_time} seconds")

                except Exception as e:
                    print(f"Error merging bands: {e}")
                    logger.error(f"Error merging bands: {e}")
                    continue

                # Reproject to WGS84 for each UTM

                try:

                    reproject_time = time.time()

                    if len(utm_32614) > 0:
                        reproject_to_wgs84(f"{output_dir}/{_folder}/32614_{band}.tif",
                                           f"{output_dir}/{_folder}/32614_WGS84_{band}.tif", "EPSG:32614")
                    if len(utm_32615) > 0:
                        reproject_to_wgs84(f"{output_dir}/{_folder}/32615_{band}.tif",
                                           f"{output_dir}/{_folder}/32615_WGS84_{band}.tif", "EPSG:32615")
                    if len(utm_32616) > 0:
                        reproject_to_wgs84(f"{output_dir}/{_folder}/32616_{band}.tif",
                                           f"{output_dir}/{_folder}/32616_WGS84_{band}.tif", "EPSG:32616")
                    # reproject_to_wgs84(f"{output_dir}/{_folder}/32614_{band}.tif", f"{output_dir}/{_folder}/32614_WGS84_{band}.tif", "EPSG:32614")
                    # reproject_to_wgs84(f"{output_dir}/{_folder}/32615_{band}.tif", f"{output_dir}/{_folder}/32615_WGS84_{band}.tif", "EPSG:32615")
                    # reproject_to_wgs84(f"{output_dir}/{_folder}/32616_{band}.tif", f"{output_dir}/{_folder}/32616_WGS84_{band}.tif", "EPSG:32616")

                    print(
                        f"Reprojection complete! Time taken: {time.time() - reproject_time} seconds")

                except Exception as e:
                    print(f"Error reprojecting bands: {e}")
                    logger.error(f"Error reprojecting bands: {e}")
                    continue

                # Store the files to merge
                # Create folder if it does not exist
                os.makedirs(f"./final_s1_IL/{_folder}", exist_ok=True)

                files_to_merge = []

                grand_merge_time = time.time()

                if len(utm_32614) > 0:
                    files_to_merge.append(
                        f"./{output_dir}/{_folder}/32614_WGS84_{band}.tif")
                if len(utm_32615) > 0:
                    files_to_merge.append(
                        f"{output_dir}/{_folder}/32615_WGS84_{band}.tif")
                if len(utm_32616) > 0:
                    files_to_merge.append(
                        f"{output_dir}/{_folder}/32616_WGS84_{band}.tif")

                # files_to_merge = [f"./{output_dir}/{_folder}/32614_WGS84_{band}.tif", f"{output_dir}/{_folder}/32615_WGS84_{band}.tif", f"{output_dir}/{_folder}/32616_WGS84_{band}.tif"]

                # Merge both UTMs in WGS84
                if len(files_to_merge) > 0:
                    merge_files(files_to_merge, "4326", band,
                                f"./final_s1_IL/{_folder}")

                    print(
                        f"Grand merge complete! Time taken: {time.time() - grand_merge_time} seconds")

                # Remove intermediate files

                remove_time = time.time()

                if len(utm_32614) > 0:
                    os.remove(f"{output_dir}/{_folder}/32614_{band}.tif")
                    os.remove(f"{output_dir}/{_folder}/32614_WGS84_{band}.tif")
                if len(utm_32615) > 0:
                    os.remove(f"{output_dir}/{_folder}/32615_{band}.tif")
                    os.remove(f"{output_dir}/{_folder}/32615_WGS84_{band}.tif")
                if len(utm_32616) > 0:
                    os.remove(f"{output_dir}/{_folder}/32616_{band}.tif")
                    os.remove(f"{output_dir}/{_folder}/32616_WGS84_{band}.tif")

                print(
                    f"Removed intermediate files for band: {band}. Time taken: {time.time() - remove_time} seconds")

            print(
                f"Processing complete for band: {band}. Time taken: {time.time() - band_processing_time} seconds")

    end_time = time.time()
    print(
        f"Processing complete! Total time taken: {end_time - start_time} seconds")
    logger.info(
        f"Processing complete! Total time taken: {end_time - start_time} seconds")
