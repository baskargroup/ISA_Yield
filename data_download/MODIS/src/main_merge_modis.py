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


def project_tif(input_tif, output_tif, target_resolution=125):
    # Open original dataset
    with rasterio.open(input_tif) as src:
        # Define new CRS
        dst_crs = "EPSG:4326"  # WGS 84

        # Compute transform
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_resolution
        )

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        # Reproject and save
        with rasterio.open(output_tif, "w", **out_meta) as dst:
            for i in range(1, src.count + 1):  # Loop through all bands
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

    print(f"Reprojected image saved at {target_resolution} = : {output_tif}")
    
    
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
    
    #Set up the logging
    my_logger = setup_logger(f"./logs/modis.log")
    
    #Get the list of directories in the base directory
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Total number of directories: {len(dirs)}")
    
    #Make the output directory if it does not exist
    for _dir in dirs:
        if not os.path.exists(f"{output_dir}/{_dir}"):
            os.makedirs(f"{output_dir}/{_dir}")
    
    for d in tqdm(dirs):
        #Project the merged tiff files to WGS 84
        for _band in define_bands[product_name]:
            tiff_files = glob.glob(f"{base_dir}/{d}/{_band}.tif")
            if len(tiff_files) == 0:
                print(f"No tiff files found for band: {_band}")
                my_logger.info(f"No tiff files found for band: {_band}")
                continue
            write_path = f"{output_dir}/{d}/{_band}_WGS84.tif"
            try:
                project_tif(tiff_files[0], write_path)
            except Exception as e:
                my_logger.error(f"Failed to project tiff files for band: {_band}. Error: {e}")
                continue
     
    end = time.time()
    my_logger.info(f"For product: {product_name}, Time taken: {end-start} seconds")   
    my_logger.info(f"For product: {product_name}, Completed processing all directories...")
    
    
    
    
                