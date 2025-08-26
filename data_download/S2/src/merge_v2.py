from logging_helper import setup_logger
#from utilities import reproject_to_wgs84, resample_to_new_resolution
from tqdm import tqdm
import time
import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
import time
import xarray as xr
import rioxarray as rxr
import os
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np


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
    logger = setup_logger("./logs/merge_v2.log")

    # Base directory
    base_dir = "../gis-stac/IA_sentinel2"
    
    #Output directory
    output_dir = "./final_s2_v2"
    
    
    temp_base_dir = "2019-07-28"
    
    # Bands to merge
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    
    # Create a directory to store the merged bands
    os.makedirs(f"{output_dir}/{temp_base_dir}", exist_ok=True)
    
    for band in bands:
        print("\n ==============================")
        print("\nProcessing band: ", band)
        print("\n ==============================")
        
        """
        Maintain 2 list per UTM
        """
        
        utm_32614 = []
        utm_32615 = []
        
        
        band_files = glob.glob(os.path.join(base_dir, temp_base_dir, f"*_{band}.tif")) #Point to source files for that week
        
        if not band_files:
            print(f"No files found for band: {band}")
            continue
        
        # Resample to 125m at Native CRS
        else:
            for file in band_files:
                print(f"Merging {file}")
                with rasterio.open(file) as src:
                    
                    if src.crs.to_string() == "EPSG:32614":
                        
                        #Resample to 125m
                        resample_to_new_resolution(file, f"{output_dir}/{temp_base_dir}/{src.crs}_R125_{os.path.basename(file)}", src.crs ,125)
                        utm_32614.append(f"{output_dir}/{temp_base_dir}/{src.crs}_R125_{os.path.basename(file)}")
                        
                    elif src.crs.to_string() == "EPSG:32615":
                        
                        #Resample to 125m
                        resample_to_new_resolution(file, f"{output_dir}/{temp_base_dir}/{src.crs}_R125_{os.path.basename(file)}", src.crs ,125)
                        utm_32615.append(f"{output_dir}/{temp_base_dir}/{src.crs}_R125_{os.path.basename(file)}")
                        
                    else:
                        print("CRS not supported")
                        continue
                    
            # Merge the bands for each UTM
            merge_files(utm_32614, "32614", band, f"{output_dir}/{temp_base_dir}")
            merge_files(utm_32615, "32615", band, f"{output_dir}/{temp_base_dir}")
            
            
            # Reproject to WGS84 for each UTM
            reproject_to_wgs84(f"{output_dir}/{temp_base_dir}/32614_{band}.tif", f"{output_dir}/{temp_base_dir}/32614_WGS84_{band}.tif", "EPSG:32614")
            reproject_to_wgs84(f"{output_dir}/{temp_base_dir}/32615_{band}.tif", f"{output_dir}/{temp_base_dir}/32615_WGS84_{band}.tif", "EPSG:32615")
            
            #Store the files to merge
            files_to_merge = [f"{output_dir}/{temp_base_dir}/32614_WGS84_{band}.tif", f"{output_dir}/{temp_base_dir}/32615_WGS84_{band}.tif"]
            
            # Merge both UTMs in WGS84
            merge_files(files_to_merge, "4326", band, f"{output_dir}/{temp_base_dir}")
            
            # Remove intermediate files
            os.remove(f"{output_dir}/{temp_base_dir}/32614_{band}.tif")
            os.remove(f"{output_dir}/{temp_base_dir}/32615_{band}.tif")
            os.remove(f"{output_dir}/{temp_base_dir}/32614_WGS84_{band}.tif")
            os.remove(f"{output_dir}/{temp_base_dir}/32615_WGS84_{band}.tif")
            
            print(f"Intermediate files removed for band: {band}")
            
        
    print("All bands merged successfully\n")
    print("Process complete!\n")      
                  
                
              