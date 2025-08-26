# Check the crs of the files
import os
import rasterio


if __name__ == "__main__":
    
    base_folder = "/work/mech-ai-scratch/rtali/gis-stac/IA_modis_NBAR"
    store_crs = []
    #For each directory in the base folder
    for d in os.listdir(base_folder):
        #For each file in the directory
        for f in os.listdir(f"{base_folder}/{d}"):
            #Check the CRS of the file
            with rasterio.open(f"{base_folder}/{d}/{f}") as src:
                store_crs.append(src.crs)

    print(f"Unique CRS: {set(store_crs)}")
    print(f"Total number of files: {len(store_crs)}")