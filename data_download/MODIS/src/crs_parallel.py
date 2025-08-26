import os
import rasterio
from rasterio.crs import CRS
from concurrent.futures import ThreadPoolExecutor


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


def get_all_tif_files(base_dir):
    """
    Get all .tif files in the directory structure.
    """
    tif_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tif'):
                tif_files.append(os.path.join(root, file))
    return tif_files


def get_all_crs_from_directory_parallel(base_dir):
    """
    Get all unique CRS from all .tif files in the directory structure using parallel processing.
    """
    tif_files = get_all_tif_files(base_dir)
    crs_set = set()
    with ThreadPoolExecutor() as executor:
        results = executor.map(get_crs_from_tif, tif_files)
        for crs in results:
            if crs is not None:
                crs_set.add(crs)
    return crs_set


def main():
    # Replace with your base directory
    base_dir = '/work/mech-ai-scratch/rtali/gis-modis-IL/IL_modis_NBAR_v2'
    unique_crs = get_all_crs_from_directory_parallel(base_dir)
    print("Unique CRS:")
    for crs in unique_crs:
        print(crs)


if __name__ == "__main__":
    main()
