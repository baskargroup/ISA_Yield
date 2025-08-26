"""
I have a directory structure like this:

base/
    ├── [date]
    │   ├── file1.tif
    │   ├── file2.tif
    │   └── file3.tif  
    ├── [date]
    │   ├── file1.tif
    │   ├── file2.tif
    │   └── file3.tif

I want to find out the CRS of each file in the directory structure and find all unique CRS.

"""

import os
import rasterio
from rasterio.crs import CRS


def get_crs_from_tif(file_path):
    """
    Get the CRS of a .tif file.
    """
    try:
        with rasterio.open(file_path) as src:
            crs = src.crs
        return crs
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def get_all_crs_from_directory(base_dir):
    """
    Get all unique CRS from all .tif files in the directory structure.
    """
    crs_set = set()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(root, file)
                crs = get_crs_from_tif(file_path)
                crs_set.add(crs)
    return crs_set


def main():
    # Replace with your base directory
    base_dir = '/work/mech-ai-scratch/rtali/gis-sentinel2-IL/IL-sentinel2-fromANVIL/IL_sentinel2'
    unique_crs = get_all_crs_from_directory(base_dir)
    print("Unique CRS:")
    for crs in unique_crs:
        print(crs)


if __name__ == "__main__":
    main()
