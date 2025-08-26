import os
import glob
import rasterio
import xarray as xr
import numpy as np
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import time
from logging_helper import setup_logger
from tqdm import tqdm
import pandas as pd
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import pyproj
import datetime
from datetime import datetime

def convert_to_julian(dt):
    """
    Convert a date to julian day.
    """
    return dt.strftime("%Y%j")

def julian_to_date(julian):
    """
    Convert a julian day to date.
    """
    return datetime.strptime(julian, "%Y%j").date()

def check_date(file_name, dt):
    file_name = os.path.basename(file_name)
    _date = file_name.split(".")[1]
    _date = _date.split("A")[1]
    _date = julian_to_date(_date)
    
    print(f"Date: {_date}")
    
    #Convert to datetime. _date is a string in the format YYYY-MM-DD
    _date = pd.to_datetime(_date, format="%Y-%m-%d")
    
    #Convert dt to datetime. dt is in the format YYYY-MM-DD
    dt = pd.to_datetime(str(dt))
    
    #Check if the _date is within 7 days of the dt
    if np.abs((_date - dt).days) <= 7:
        return True
    else:
        return False



def clip_raster(input_tif, gdf, output_tif):
    with rasterio.open(input_tif) as src:
        geom = [gdf.geometry[0].__geo_interface__]  # Convert to GeoJSON format
        out_image, out_transform = mask(src, geom, crop=True)
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Save clipped raster
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)
            

def resample_to_new_resolution(input_tif, output_tif, src_crs , target_resolution=125):
    with rasterio.open(input_tif) as src:
        # Define the target CRS. Since we are resampling, the target CRS is the same as the source CRS
        dst_crs = src_crs

        # Compute the transform, width, and height for the new resolution
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_resolution
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

def reproject_raster(input_tif, output_tif, target_crs):
    with rasterio.open(input_tif) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        out_meta = src.meta.copy()
        out_meta.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })
        
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dest, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs
                )
            

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
            
    print(f"Conversion complete! Data reprojected to WGS84 and saved as {output_path}")

def merge_files(list_of_files, crs , band_name, output_dir):
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
    logger = setup_logger("./logs/modis_reproject_local.log")
    
    # Base directory
    base_dir_pattern = "../gis-stac/IA_modis_NBAR/2023-*"
    base_dirs = glob.glob(base_dir_pattern)
    
    #Sort the directories
    base_dirs.sort()
    
    #Output directory
    output_dir = "./reprojected_modis"
    
    # a = os.listdir("../gis-stac/IA_sentinel1")
    # b = os.listdir("./process_s1")
    # diff = dir_diff(a, b)
    
    # Bands to merge
    bands = ['Band1','Band2','Band3','Band4','Band5','Band6','Band7']
    
    # Go to each directory within the base directory and merge the bands. There are several bands
    
    start_time = time.time()
    
    print(f"Start Time in CDT: {time.ctime(start_time)}")
    
    for folder in tqdm(base_dirs, desc="Processing folders"):
    # for folder in diff:
        
        #Extract the folder name from the path
        _folder = folder.split("/")[-1]
        
        os.makedirs(f"{output_dir}/{_folder}", exist_ok=True)

        
        print(f"Processing folder: {folder}")
        
        # Get the list of files for each band
        
        for band in bands:
            
            band_processing_time = time.time()
            
            """
            Maintain 2 list per UTM
            """
            
            # Get the list of files for each band
            band_files = glob.glob(os.path.join(folder, f"*_{band}.tif"))
            
            # Check that dates from the file names are between _folder and _folder + 7 days
            # If not, remove the file from the list
            band_files = [file for file in band_files if check_date(file, _folder)]
            
            if not band_files:
                print(f"No files found for band: {band}")
                logger.info(f"No files found for band: {band}")
                continue
    
            else:
                for file in band_files:
                    with rasterio.open(file) as src:
                       # Step - 1 : Reprohect the file to WGS84 
                       reproject_to_wgs84(file, f"{output_dir}/{_folder}/{os.path.basename(file)}", src.crs)
                       
                    
                    # Step - 2 : Split the file into UTM zones
                    with rasterio.open(f"{output_dir}/{_folder}/{os.path.basename(file)}") as src:
                        
                        bounds = src.bounds  # (left, bottom, right, top)
                        crs = src.crs
                        print(f"Original CRS: {crs}, Bounds: {bounds}")
                        
                        # Define bounding boxes for each UTM zone
                        zone14_bbox = box(-96.6395, 40.3754, -96, 43.5014)
                        zone15_bbox = box(-96, 40.3754, -90.1401, 43.5014)

                        # Create GeoDataFrames
                        gdf_zone14 = gpd.GeoDataFrame({"geometry": [zone14_bbox]}, crs="EPSG:4326")
                        gdf_zone15 = gpd.GeoDataFrame({"geometry": [zone15_bbox]}, crs="EPSG:4326")
                        
                        # Clip into two parts
                        clip_raster(f"{output_dir}/{_folder}/{os.path.basename(file)}", gdf_zone14, f"{output_dir}/{_folder}/CUT_32614_{os.path.basename(file)}")
                        clip_raster(f"{output_dir}/{_folder}/{os.path.basename(file)}", gdf_zone15, f"{output_dir}/{_folder}/CUT_32615_{os.path.basename(file)}")
                        
                        # Reproject to UTM zones
                        # Reproject to UTM zones
                        reproject_raster(f"{output_dir}/{_folder}/CUT_32614_{os.path.basename(file)}", f"{output_dir}/{_folder}/32614_{os.path.basename(file)}", "EPSG:32614")
                        reproject_raster(f"{output_dir}/{_folder}/CUT_32615_{os.path.basename(file)}", f"{output_dir}/{_folder}/32615_{os.path.basename(file)}", "EPSG:32615")
                        
                        # Resample to 125m
                        resample_to_new_resolution(f"{output_dir}/{_folder}/32614_{os.path.basename(file)}", f"{output_dir}/{_folder}/R125_32614_{os.path.basename(file)}", "EPSG:32614")
                        resample_to_new_resolution(f"{output_dir}/{_folder}/32615_{os.path.basename(file)}", f"{output_dir}/{_folder}/R125_32615_{os.path.basename(file)}", "EPSG:32615")
                        
                        # Reproject to WGS84
                        reproject_to_wgs84(f"{output_dir}/{_folder}/R125_32614_{os.path.basename(file)}", f"{output_dir}/{_folder}/WGS84_32614_{os.path.basename(file)}", "EPSG:32614")
                        reproject_to_wgs84(f"{output_dir}/{_folder}/R125_32615_{os.path.basename(file)}", f"{output_dir}/{_folder}/WGS84_32615_{os.path.basename(file)}", "EPSG:32615")
                        
                        # Merge the reprojected files into one
                        wgs_files = [f"{output_dir}/{_folder}/WGS84_32614_{os.path.basename(file)}", f"{output_dir}/{_folder}/WGS84_32615_{os.path.basename(file)}"]
                        
                        #Create a folder for the band
                        os.makedirs(f"./final_modis_data/{_folder}", exist_ok=True)
                        write_file_name = os.path.basename(file).replace(".tif", "").strip()
                        merge_files(wgs_files, f"4326_{write_file_name}", band, f"./final_modis_data/{_folder}")
                        
            print(f"Processing time for band {band}: {time.time() - band_processing_time}")
                         
    print(f"Total processing time: {time.time() - start_time}")
    print(f"End Time in CDT: {time.ctime(time.time())}")
                        
                        