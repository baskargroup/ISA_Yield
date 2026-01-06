import os
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import box
import glob
import numpy as np
from datetime import datetime
import gc
import xarray as xr
import pdb
# Define the paths
ext = "IA"
years = [2019,2020,2021,2022,2023]
modalities = [#'S2L2A', 
              #'S1RTC',
            #   'MODIS',
              'DEM',
            #   'CDL',
            #   'WEATHER',
            #   'SOIL',
              ]
# sentinel1_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/data_download/S1/final_s1_{ext}/'
# sentinel2_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/data_download/S2/final_s2_v3_{ext}/'
# modis_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/data_download/MODIS/modis_{ext}/'
crop_dir = '/work/mech-ai-scratch/aapowadi/multimodal_fusion/remapped_cdl'
soil_dir = f'/work/mech-ai-scratch/aapowadi/soil_new/soil_processed_{ext}'
weather_dir = f'/work/mech-ai-scratch/aapowadi/WEEKLY_WEATHER_{ext}'
dem_path = f'/work/mech-ai-scratch/bgekim/project/ISA_Yield/yld_proc/unprocessed_data/dem_iowa_10m_utm.tif'
output_dir = f'/work/mech-ai-scratch/bgekim/project/ISA_Yield/chloe_dataset'
yield_path = '/work/mech-ai-scratch/bgekim/project/ISA_Yield/yld_proc/unprocessed_data/yield_geotiffs'


# Ensure output directories exist
for m in modalities:
    os.makedirs(os.path.join(output_dir, m), exist_ok=True)

# Get list of yield geotiff files for all selected years using the naming convention STYYYYIA*_*.npy
yield_files = []
for year in years:
    yield_files.extend(glob.glob(os.path.join(yield_path, f'ST{year}IA*_*.tif')))
    
# ** Chloe **
def get_bbox_from_geotiff(file_path):
    """Extract the bounding box from a geotiff file."""
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        crs = src.crs
    return gpd.GeoDataFrame({'geometry': [bbox]}, crs=crs)


# ** Chloe **
def is_valid_date_combined(input_file, bbox_gdf, modality):
    """Check if a pre-combined file date is valid."""
    # ✅ 이미 combined된 파일 하나만 읽음
    # input_file = 'sentinel2_dir/2023-04-01.tif' (12 bands)
    
    try:
        with rasterio.open(input_file) as src:
            bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
            out_image, _ = mask(
                src,
                bbox_gdf_reprojected.geometry,
                crop=True
            )
        
        if out_image is None:
            return False
        
        # ✅ 유효성 검사 (기존과 동일)
        if modality != 'WEATHER':
            valid_per_band = ~np.isnan(out_image) & (out_image != 0)
        else:
            valid_per_band = ~np.isnan(out_image)
        
        all_valid = np.all(valid_per_band, axis=0)
        frac_valid = np.mean(all_valid)
        
        del out_image, valid_per_band, all_valid
        gc.collect()
        
        return frac_valid > 0.7
        
    except Exception as e:
        print(f"Error checking validity for {input_file}: {e}")
        return False


# ** Chloe **
def clip_geotiff(input_file, output_path, bbox_gdf, repeat_times=1):
    """Clip pre-combined multi-band geotiff using geospatial windowed reading."""
    try:
        with rasterio.open(input_file) as src:
            # bbox를 raster CRS로 변환
            bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
            
            # clip
            out_image, out_transform = mask(
                src,
                bbox_gdf_reprojected.geometry,
                crop=True
            )
        
        # repeat이 필요한 경우 (예: static data를 시계열로 확장)
        if repeat_times > 1:
            out_image = np.tile(out_image, (repeat_times, 1, 1))
        
        # Save as .npy
        np.save(output_path, out_image, allow_pickle=False)
        
        del out_image
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# ** Chloe **
def clip_multidate_geotiff(input_dir, output_path, bbox_gdf, selected_dates, modality):
    """Clip multiple pre-combined multi-band geotiff files and stack by date."""
    try:
        clipped_arrays = []
        
        for date in selected_dates:
            # 파일 경로 (이미 band가 combined된 파일)
            input_file = os.path.join(input_dir, f'{date}.tif')
            
            if not os.path.exists(input_file):
                print(f"Warning: {input_file} not found, skipping")
                continue
            
            # ✅ Clip만 수행 (combine 없음!)
            with rasterio.open(input_file) as src:
                bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
                out_image, out_transform = mask(
                    src,
                    bbox_gdf_reprojected.geometry,
                    crop=True
                )
                clipped_arrays.append(out_image)
        
        if not clipped_arrays:
            print(f"No valid files found for {output_path}")
            return
        
        # ✅ 날짜별로 stack만 (band combine 아님!)
        stacked = np.concatenate(clipped_arrays, axis=0)
        
        # Reshape to (dates, bands, H, W)
        num_dates = len(selected_dates)
        num_bands = clipped_arrays[0].shape[0]
        height = clipped_arrays[0].shape[1]
        width = clipped_arrays[0].shape[2]
        out_arr = np.reshape(stacked, (num_dates, num_bands, height, width))
        
        np.save(output_path, out_arr, allow_pickle=False)
        
        del clipped_arrays, stacked, out_arr
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")

# Define dynamic and static modalities
dynamic_mods = [#'S1RTC', 
                # 'S2L2A',
                # 'MODIS',
                # 'WEATHER',
                ]
mod_to_dir = {
    # 'S1RTC': sentinel1_dir,
    # 'S2L2A': sentinel2_dir,
    # 'MODIS': modis_dir,
    'WEATHER': weather_dir,
    'SOIL': soil_dir,
    'CDL': crop_dir,
    'DEM': dem_path
}


# ** Chloe ** ✅ Process
log_file = os.path.join(output_dir, "used_dates_log.txt")
with open(log_file, "w") as logf:
    for yield_file in yield_files:
        base_name = os.path.basename(yield_file)
        year = base_name.split('IA')[0][-4:]
        print(f"Processing yield file: {base_name} for year {year}")
        bbox_gdf = get_bbox_from_geotiff(yield_file)
        
        # Dynamic modalities
        if dynamic_mods:
            valid_dates = {}
            
            for mod in dynamic_mods:
                input_dir = mod_to_dir[mod]
                
                if mod == 'WEATHER':
                    files = os.listdir(input_dir)
                    possible_dates = sorted(set(f.replace('.tif', '') for f in files 
                                               if f.startswith(str(year)) and f.endswith('.tif')))
                    possible_dates = [d for d in possible_dates if 4 <= int(d.split('-')[1]) <= 9]
                else:
                    files = [f.replace('.tif', '') for f in os.listdir(input_dir) 
                            if f.endswith('.tif') and f.startswith(str(year))]
                    possible_dates = sorted(files)
                
                valids = []
                for date in possible_dates:
                    input_file = os.path.join(input_dir, f'{date}.tif')
                    if is_valid_date_combined(input_file, bbox_gdf, mod):
                        valids.append(date)
                
                valid_dates[mod] = valids
                print(f"Number of valid dates for {mod}: {len(valid_dates[mod])}")
            
            if all(len(valid_dates[mod]) == 0 for mod in dynamic_mods):
                print(f"No valid dates for any modality for {base_name}, skipping.")
                continue
            
            sets_of_dates = [set(valid_dates[mod]) for mod in dynamic_mods if len(valid_dates[mod]) > 0]
            if not sets_of_dates or len(sets_of_dates) < len(dynamic_mods):
                print(f"Not all modalities have valid dates for {base_name}, skipping.")
                continue
            
            common_dates = sorted(set.intersection(*sets_of_dates))
            if not common_dates:
                print(f"No common dates across all modalities for {base_name}, skipping.")
                continue
            
            selected_dates = {mod: common_dates for mod in dynamic_mods}
            logf.write(f"{base_name}: {','.join(common_dates)}\n")
            
            for mod in dynamic_mods:
                if mod not in selected_dates:
                    continue
                output_path = os.path.join(output_dir, mod, base_name.replace('.tif', '.npy'))
                clip_multidate_geotiff(mod_to_dir[mod], output_path, bbox_gdf, selected_dates[mod], mod)
                print(f"✅ Processed {mod} for {base_name}")
        
        # Static modalities
        num_times = 12
        
        for mod in ['DEM']:  # Change to ['DEM', 'SOIL', 'CDL'] when ready
            output_path = os.path.join(output_dir, mod, base_name.replace('.tif', '.npy'))
            
            if mod == 'CDL':
                input_path = os.path.join(mod_to_dir[mod], f'{year}_{ext}_CDL.tif')
            elif mod == 'DEM':
                input_path = dem_path
            else:  # SOIL
                input_path = mod_to_dir[mod]
            
            clip_geotiff(input_path, output_path, bbox_gdf, repeat_times=num_times)
            print(f"✅ Processed {mod} for {base_name}")
        
        gc.collect()

print("\n🎉 Processing complete!")

# ## ---------------------------------------- - save for tif -----------------------------------------

# import os
# import rasterio
# from rasterio.mask import mask
# import geopandas as gpd
# from shapely.geometry import box
# import glob
# import numpy as np
# import gc

# # Define the paths
# ext = "IA"
# years = [2019, 2020, 2021, 2022, 2023]
# modalities = ['DEM']  # 'SOIL', 'CDL' 추가 가능

# crop_dir = '/work/mech-ai-scratch/aapowadi/multimodal_fusion/remapped_cdl'
# soil_dir = f'/work/mech-ai-scratch/aapowadi/soil_new/soil_processed_{ext}'
# weather_dir = f'/work/mech-ai-scratch/aapowadi/WEEKLY_WEATHER_{ext}'
# dem_path = f'/work/mech-ai-scratch/bgekim/project/ISA_Yield/yld_proc/unprocessed_data/dem_iowa_10m_utm.tif'
# output_dir = f'/work/mech-ai-scratch/bgekim/project/ISA_Yield/chloe_dataset'
# yield_path = '/work/mech-ai-scratch/bgekim/project/ISA_Yield/yld_proc/unprocessed_data/yield_geotiffs'

# # Ensure output directories exist
# for m in modalities:
#     os.makedirs(os.path.join(output_dir, m), exist_ok=True)

# # Get list of yield geotiff files
# yield_files = []
# for year in years:
#     yield_files.extend(glob.glob(os.path.join(yield_path, f'ST{year}IA*_*.tif')))

# # ========== Functions ==========

# def get_bbox_from_geotiff(file_path):
#     """Extract the bounding box from a geotiff file."""
#     with rasterio.open(file_path) as src:
#         bounds = src.bounds
#         bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
#         crs = src.crs
#     return gpd.GeoDataFrame({'geometry': [bbox]}, crs=crs)


# def is_valid_date_combined(input_file, bbox_gdf, modality):
#     """Check if a pre-combined file date is valid."""
#     try:
#         with rasterio.open(input_file) as src:
#             bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
#             out_image, _ = mask(src, bbox_gdf_reprojected.geometry, crop=True)
        
#         if out_image is None:
#             return False
        
#         if modality != 'WEATHER':
#             valid_per_band = ~np.isnan(out_image) & (out_image != 0)
#         else:
#             valid_per_band = ~np.isnan(out_image)
        
#         all_valid = np.all(valid_per_band, axis=0)
#         frac_valid = np.mean(all_valid)
        
#         del out_image, valid_per_band, all_valid
#         gc.collect()
        
#         return frac_valid > 0.7
        
#     except Exception as e:
#         print(f"Error checking validity for {input_file}: {e}")
#         return False


# def clip_geotiff(input_file, output_path, bbox_gdf, repeat_times=1):
#     """Clip pre-combined multi-band geotiff and save as GeoTIFF."""
#     try:
#         with rasterio.open(input_file) as src:
#             bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
            
#             # Clip
#             out_image, out_transform = mask(
#                 src,
#                 bbox_gdf_reprojected.geometry,
#                 crop=True
#             )
            
#             # Get metadata
#             out_meta = src.meta.copy()
        
#         # Repeat if needed (static data → temporal)
#         if repeat_times > 1:
#             out_image = np.tile(out_image, (repeat_times, 1, 1))
        
#         # Update metadata
#         out_meta.update({
#             "driver": "GTiff",
#             "height": out_image.shape[1],
#             "width": out_image.shape[2],
#             "count": out_image.shape[0],
#             "transform": out_transform
#         })
        
#         # ✅ Save as GeoTIFF
#         with rasterio.open(output_path, "w", **out_meta) as dest:
#             dest.write(out_image)
        
#         del out_image
#         gc.collect()
        
#     except Exception as e:
#         print(f"Error processing {input_file}: {e}")


# def clip_multidate_geotiff(input_dir, output_path, bbox_gdf, selected_dates, modality):
#     """Clip multiple pre-combined multi-band geotiff files, stack by date, and save as GeoTIFF."""
#     try:
#         clipped_arrays = []
#         out_transform = None
#         out_crs = None
#         out_dtype = None
        
#         for date in selected_dates:
#             input_file = os.path.join(input_dir, f'{date}.tif')
            
#             if not os.path.exists(input_file):
#                 print(f"Warning: {input_file} not found, skipping")
#                 continue
            
#             # Clip
#             with rasterio.open(input_file) as src:
#                 bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
#                 out_image, out_transform = mask(
#                     src,
#                     bbox_gdf_reprojected.geometry,
#                     crop=True
#                 )
#                 clipped_arrays.append(out_image)
                
#                 # Save metadata from first file
#                 if out_crs is None:
#                     out_crs = src.crs
#                     out_dtype = src.dtypes[0]
        
#         if not clipped_arrays:
#             print(f"No valid files found for {output_path}")
#             return
        
#         # Stack dates
#         stacked = np.concatenate(clipped_arrays, axis=0)
        
#         # Get dimensions
#         num_dates = len(selected_dates)
#         num_bands = clipped_arrays[0].shape[0]
#         height = clipped_arrays[0].shape[1]
#         width = clipped_arrays[0].shape[2]
        
#         # Reshape: (dates, bands, H, W) → (dates*bands, H, W)
#         # GeoTIFF format: (bands, H, W)
#         out_arr = np.reshape(stacked, (num_dates * num_bands, height, width))
        
#         # ✅ Save as GeoTIFF
#         with rasterio.open(
#             output_path,
#             'w',
#             driver='GTiff',
#             height=height,
#             width=width,
#             count=num_dates * num_bands,
#             dtype=out_dtype,
#             crs=out_crs,
#             transform=out_transform
#         ) as dest:
#             dest.write(out_arr)
        
#         print(f"  → Saved: {num_dates} dates × {num_bands} bands = {num_dates*num_bands} total bands")
        
#         del clipped_arrays, stacked, out_arr
#         gc.collect()
        
#     except Exception as e:
#         print(f"Error processing {input_dir}: {e}")


# # ========== Define Modalities ==========

# dynamic_mods = []  # Add ['S2L2A', 'S1RTC', 'WEATHER'] when ready

# mod_to_dir = {
#     # 'S1RTC': sentinel1_dir,
#     # 'S2L2A': sentinel2_dir,
#     'WEATHER': weather_dir,
#     'SOIL': soil_dir,
#     'CDL': crop_dir,
#     'DEM': dem_path
# }


# # ========== Process ==========

# log_file = os.path.join(output_dir, "used_dates_log.txt")
# with open(log_file, "w") as logf:
#     for yield_file in yield_files:
#         base_name = os.path.basename(yield_file)
#         year = base_name.split('IA')[0][-4:]
#         print(f"Processing yield file: {base_name} for year {year}")
#         bbox_gdf = get_bbox_from_geotiff(yield_file)
        
#         # ========== Dynamic Modalities ==========
#         if dynamic_mods:
#             valid_dates = {}
            
#             for mod in dynamic_mods:
#                 input_dir = mod_to_dir[mod]
                
#                 if mod == 'WEATHER':
#                     files = os.listdir(input_dir)
#                     possible_dates = sorted(set(f.replace('.tif', '') for f in files 
#                                                if f.startswith(str(year)) and f.endswith('.tif')))
#                     possible_dates = [d for d in possible_dates if 4 <= int(d.split('-')[1]) <= 9]
#                 else:
#                     files = [f.replace('.tif', '') for f in os.listdir(input_dir) 
#                             if f.endswith('.tif') and f.startswith(str(year))]
#                     possible_dates = sorted(files)
                
#                 valids = []
#                 for date in possible_dates:
#                     input_file = os.path.join(input_dir, f'{date}.tif')
#                     if is_valid_date_combined(input_file, bbox_gdf, mod):
#                         valids.append(date)
                
#                 valid_dates[mod] = valids
#                 print(f"Number of valid dates for {mod}: {len(valid_dates[mod])}")
            
#             if all(len(valid_dates[mod]) == 0 for mod in dynamic_mods):
#                 print(f"No valid dates for any modality for {base_name}, skipping.")
#                 continue
            
#             sets_of_dates = [set(valid_dates[mod]) for mod in dynamic_mods if len(valid_dates[mod]) > 0]
#             if not sets_of_dates or len(sets_of_dates) < len(dynamic_mods):
#                 print(f"Not all modalities have valid dates for {base_name}, skipping.")
#                 continue
            
#             common_dates = sorted(set.intersection(*sets_of_dates))
#             if not common_dates:
#                 print(f"No common dates across all modalities for {base_name}, skipping.")
#                 continue
            
#             selected_dates = {mod: common_dates for mod in dynamic_mods}
#             logf.write(f"{base_name}: {','.join(common_dates)}\n")
            
#             for mod in dynamic_mods:
#                 if mod not in selected_dates:
#                     continue
#                 # ✅ Save as .tif (not .npy!)
#                 output_path = os.path.join(output_dir, mod, base_name)
#                 clip_multidate_geotiff(mod_to_dir[mod], output_path, bbox_gdf, selected_dates[mod], mod)
#                 print(f"✅ Processed {mod} for {base_name}")
        
#         # ========== Static Modalities ==========
#         num_times = 12
        
#         for mod in ['DEM']:  # Change to ['DEM', 'SOIL', 'CDL'] when ready
#             # ✅ Save as .tif (not .npy!)
#             output_path = os.path.join(output_dir, mod, base_name)
            
#             if mod == 'CDL':
#                 input_path = os.path.join(mod_to_dir[mod], f'{year}_{ext}_CDL.tif')
#             elif mod == 'DEM':
#                 input_path = dem_path
#             else:  # SOIL
#                 input_path = mod_to_dir[mod]
            
#             clip_geotiff(input_path, output_path, bbox_gdf, repeat_times=num_times)
#             print(f"✅ Processed {mod} for {base_name}")
        
#         gc.collect()

# print("\n🎉 Processing complete!")
# print(f"Output directory: {output_dir}")
# print("All files saved as GeoTIFF (.tif)")