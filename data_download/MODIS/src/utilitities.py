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
#from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np


def stack_bands_to_netcdf(tif_files, output_file_name = "./processed_soil/output.nc"):
    # Load all TIFFs into a list of xarray DataArrays
    bands = [rxr.open_rasterio(f).squeeze() for f in tif_files]

    # Stack along a new "band" dimension
    stacked = xr.concat(bands, dim="band")
    
    # Save to NetCDF
    encoding = {"band": {"zlib": True, "complevel": 5}}
    stacked.to_netcdf(output_file_name, engine="netcdf4", encoding=encoding)

    print(f"NetCDF file saved as {output_file_name}")
    
    
# Plot the preview images one by one
def view_tif_files(preview_files, output_folder = "./plots/"):
    for file in preview_files:
        try:
            with rasterio.open(file) as src:
                show(src)
                #Extract the filename without the extension and the path
                file_name = os.path.splitext(os.path.basename(file))[0]
                print(f"Plotting {file_name}")
                #Save the plot as a png file
                plt.savefig(f"{output_folder}/{file_name}.png")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

def merge_soil_data(soil_folder, output_folder , band_name):
    # List all .tif files in the directory
    tif_files = glob.glob(f"./{soil_folder}/*_{band_name}.tif")

    # Open all TIFF files
    try:
        src_files = [rasterio.open(f) for f in tif_files]
    except Exception as e:
        print(f"Error: {e}")

    # Merge with max criterion
    mosaic, out_trans = merge(src_files, method="max")

    # Copy metadata
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    # Save merged TIFF
    output_path = f"./{output_folder}/merged_max_{band_name}.tif"
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Merged file saved as {output_path} using max value criteria.")



def reproject_to_wgs84(input_path, output_path, src_crs, dst_crs="EPSG:4326"):
    """
    Reprojects a given TIFF file to a new CRS.

    Args:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path to save the reprojected TIFF file.
        dst_crs (str): Destination CRS (default: "EPSG:4326").
    
    Returns:
        str: Path to the reprojected TIFF file.
    """
    
    src = rasterio.open(input_path)
    
    # Define source and target CRS
    # src_crs = 'ESRI:102039'
    # dst_crs = 'EPSG:32632'

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
        # Define target CRS (WGS84)
        dst_crs = src_crs

        # Compute the transform, width, and height for the new resolution
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_resolution
        )

        print(f"Original resolution: {src.res}")
        print(f"New resolution: {target_resolution}m")
        print(f"New dimensions: {width}x{height}")

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