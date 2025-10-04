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
years = [2019, 2020, 2021, 2022, 2023, 2024]
modalities = ['S2L2A', 'S1GRD', 'MODIS', 'DEM', 'CDL', 'WEATHER', 'SOIL']
sentinel1_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/data_download/S1/final_s1_{ext}/'
sentinel2_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/data_download/S2/final_s2_v3_{ext}/'
modis_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/data_download/MODIS/modis_{ext}/'
crop_dir = '/work/mech-ai-scratch/aapowadi/multimodal_fusion/remapped_cdl'
soil_dir = f'/work/mech-ai-scratch/aapowadi/soil_new/soil_processed_{ext}'
weather_dir = f'/work/mech-ai-scratch/aapowadi/WEEKLY_WEATHER_{ext}'
dem_path = f'/work/mech-ai-scratch/rtali/multimodal_fusion/terrain_merged/4326_elevation_{ext}.tif'
output_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/unprocessed_data'
yield_path = '/work/mech-ai-scratch/aapowadi/ISA_Yield/unprocessed_data/yield_geotiffs'

# Define bands for each modality
s1_bands = ['vv', 'vh']
s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
modis_bands = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7']
weather_bands = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
soil_bands = ['aws100', 'aws150', 'aws999', 'nccpi3all', 'nccpi3corn', 'rootznaws', 'soc150', 'soc999', 'pctearthmc', 'nccpi3soy']
dem_band = ['band_data']

# Ensure output directories exist
for m in modalities:
    os.makedirs(os.path.join(output_dir, m), exist_ok=True)

# Get list of yield geotiff files for all selected years using the naming convention STYYYYIA*_*.npy
yield_files = []
for year in years:
    yield_files.extend(glob.glob(os.path.join(yield_path, f'ST{year}IA*_*.tif')))
def get_bbox_from_geotiff(file_path):
    """Extract the bounding box from a geotiff file."""
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        crs = src.crs
    return gpd.GeoDataFrame({'geometry': [bbox]}, crs=crs)

def load_band(band_file, bbox_gdf, ref_shape=None, ref_transform=None, ref_crs=None):
    """Load a single band using geospatial coordinates for windowed reading."""
    with rasterio.open(band_file) as src:
        # Reproject bbox to source CRS
        bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
        bounds = bbox_gdf_reprojected.geometry.iloc[0].bounds  # (left, bottom, right, top)
        # Define window using geospatial coordinates
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            raise ValueError(f"Invalid window for {band_file}: {window}")
        data = src.read(1, window=window)
        window_transform = src.window_transform(window)
        if ref_shape is None:
            return data, (int(window.height), int(window.width)), window_transform, src.crs, src.meta.copy()
        else:
            resampled_data = np.empty(ref_shape, dtype=data.dtype)
            reproject(
                source=data,
                destination=resampled_data,
                src_transform=window_transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest
            )
            return resampled_data

def get_stacked_for_date(input_dir, bbox_gdf, band_list, date, modality):
    """Stack bands for a specific date using geospatial windowed reading."""
    try:
        ref_shape = None
        ref_transform = None
        ref_crs = None
        meta = None
        band_arrays = []
        for band in band_list:
            if modality == 'WEATHER':
                band_file = os.path.join(input_dir, f'{date}_{band}.tif')
            else:
                date_dir = os.path.join(input_dir, date)
                band_file = os.path.join(date_dir, f'4326_{band}.tif')
            if not os.path.exists(band_file):
                return None
            if ref_shape is None:
                data, ref_shape, ref_transform, ref_crs, meta = load_band(band_file, bbox_gdf)
                band_arrays.append(data)
            else:
                data = load_band(band_file, bbox_gdf, ref_shape, ref_transform, ref_crs)
                band_arrays.append(data)
            del data
            gc.collect()
        if len(band_arrays) < len(band_list):
            return None
        stacked_array = np.stack(band_arrays, axis=0)
        meta.update({'count': stacked_array.shape[0], 'height': ref_shape[0], 'width': ref_shape[1], 'transform': ref_transform})
        bbox_gdf_reprojected = bbox_gdf.to_crs(ref_crs)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(stacked_array)
                out_image, out_transform = mask(
                    dataset,
                    bbox_gdf_reprojected.geometry,
                    crop=True
                )
        del stacked_array, band_arrays
        gc.collect()
        return out_image
    except Exception as e:
        print(f"Error getting stacked for date {date} in {modality}: {e}")
        return None

def is_valid_date(input_dir, bbox_gdf, band_list, date, modality):
    """Check if a date is valid based on data availability and quality."""
    out_image = get_stacked_for_date(input_dir, bbox_gdf, band_list, date, modality)
    if out_image is None:
        return False
    if modality != 'WEATHER':
        valid_per_band = ~np.isnan(out_image) & (out_image != 0)
    else:
        valid_per_band = ~np.isnan(out_image)
    all_valid = np.all(valid_per_band, axis=0)
    frac_valid = np.mean(all_valid)
    del out_image, valid_per_band, all_valid
    gc.collect()
    return frac_valid > 0.7

def combine_and_clip_geotiff(input_dir, output_path, bbox_gdf, band_list, year, modality, is_dated=False, selected_dates=None, repeat_times=1):
    """Combine and clip geotiff bands using geospatial windowed reading, save as .npy and xarray dataset."""
    try:
        if selected_dates is None:
            selected_dates = []
        ref_shape = None
        ref_transform = None
        ref_crs = None
        meta = None
        band_arrays = []
        band_names = []
        time_coords = []

        if is_dated:
            dates_to_use = selected_dates
            for date in dates_to_use:
                for band in band_list:
                    if modality == 'WEATHER':
                        band_file = os.path.join(input_dir, f'{date}_{band}.tif')
                    else:
                        date_dir = os.path.join(input_dir, date)
                        band_file = os.path.join(date_dir, f'4326_{band}.tif')
                    if not os.path.exists(band_file):
                        continue
                    if ref_shape is None:
                        data, ref_shape, ref_transform, ref_crs, meta = load_band(band_file, bbox_gdf)
                        band_arrays.append(data)
                        band_names.append(f"{band}_{date}")
                        if date not in time_coords:
                            time_coords.append(date)
                    else:
                        data = load_band(band_file, bbox_gdf, ref_shape, ref_transform, ref_crs)
                        band_arrays.append(data)
                        band_names.append(f"{band}_{date}")
                    del data
                    gc.collect()
        else:
            for band in band_list:
                if modality == 'SOIL':
                    band_file = os.path.join(input_dir, f'{band}.tif')
                elif modality == 'CDL':
                    band_file = os.path.join(input_dir, f'{year}_{ext}_CDL.tif')
                else:
                    band_file = input_dir
                if not os.path.exists(band_file):
                    continue
                if ref_shape is None:
                    data, ref_shape, ref_transform, ref_crs, meta = load_band(band_file, bbox_gdf)
                    band_arrays.append(data)
                    band_names.append(band)
                else:
                    data = load_band(band_file, bbox_gdf, ref_shape, ref_transform, ref_crs)
                    band_arrays.append(data)
                    band_names.append(band)
                del data
                gc.collect()

        if not band_arrays:
            print(f"No bands found for {input_dir}")
            return

        stacked_array = np.stack(band_arrays, axis=0)
        if not is_dated and repeat_times > 1:
            stacked_array = np.tile(stacked_array, (repeat_times, 1, 1))
            band_names = [name for name in band_names for _ in range(repeat_times)]

        meta.update({'count': stacked_array.shape[0], 'height': ref_shape[0], 'width': ref_shape[1], 'transform': ref_transform})
        bbox_gdf_reprojected = bbox_gdf.to_crs(ref_crs)

        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(stacked_array)
                out_image, out_transform = mask(
                    dataset,
                    bbox_gdf_reprojected.geometry,
                    crop=True
                )

        height = out_image.shape[1]
        width = out_image.shape[2]
        out_arr = np.reshape(out_image, (-1, len(band_list), height, width))
        # pdb.set_trace()
        # Save as .npy
        np.save(output_path, out_arr, allow_pickle=False)

        del stacked_array, out_image, out_arr, band_arrays #, ds
        gc.collect()

    except Exception as e:
        print(f"Error processing {input_dir}: {e}")

# Define dynamic and static modalities
dynamic_mods = ['S1GRD', 'S2L2A', 'MODIS', 'WEATHER']
mod_to_dir = {
    'S1GRD': sentinel1_dir,
    'S2L2A': sentinel2_dir,
    'MODIS': modis_dir,
    'WEATHER': weather_dir,
    'SOIL': soil_dir,
    'CDL': crop_dir,
    'DEM': dem_path
}
mod_to_bands = {
    'S1GRD': s1_bands,
    'S2L2A': s2_bands,
    'MODIS': modis_bands,
    'WEATHER': weather_bands,
    'SOIL': soil_bands,
    'CDL': ['CDL'],
    'DEM': dem_band
}

log_file = os.path.join(output_dir, "used_dates_log.txt")
with open(log_file, "w") as logf:
    # Process each yield file
    for yield_file in yield_files:
        base_name = os.path.basename(yield_file)
        # Extract year from filename (expects format: <something>_<year>_<something>.tif)
        year = base_name.split('IA')[0][-4:]
        print(f"Processing yield file: {base_name} for year {year}")
        bbox_gdf = get_bbox_from_geotiff(yield_file)
        valid_dates = {}
        for mod in dynamic_mods:
            input_dir = mod_to_dir[mod]
            band_list = mod_to_bands[mod]
            if mod == 'WEATHER':
                files = os.listdir(input_dir)
                possible_dates = sorted(set(f.split('_')[0] for f in files if f.startswith(str(year))))
                possible_dates = [d for d in possible_dates if 4 <= int(d.split('-')[1]) <= 9]
            else:
                possible_dates = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith(str(year) + '-')]
                possible_dates = sorted(possible_dates)
            valids = []
            for date in possible_dates:
                if is_valid_date(input_dir, bbox_gdf, band_list, date, mod):
                    valids.append(date)
            valid_dates[mod] = valids
            print(f"Number of valid dates for {mod} is {len(valid_dates[mod])}")
        if all(len(valid_dates[mod]) == 0 for mod in dynamic_mods):
            print(f"No valid dates for any modality for {base_name}, skipping.")
            continue
        # Find intersection of valid dates across all dynamic modalities
        sets_of_dates = [set(valid_dates[mod]) for mod in dynamic_mods if len(valid_dates[mod]) > 0]
        if not sets_of_dates or len(sets_of_dates) < len(dynamic_mods):
            print(f"Not all modalities have valid dates for {base_name}, skipping.")
            continue
        common_dates = sorted(set.intersection(*sets_of_dates))
        if not common_dates:
            print(f"No common dates across all modalities for {base_name}, skipping.")
            continue
        num_times = len(common_dates)
        selected_dates = {mod: common_dates for mod in dynamic_mods}
        # --- LOG THE DATES USED ---
        logf.write(f"{base_name}: {','.join(common_dates)}\n")
        # Process dynamic modalities
        for mod in dynamic_mods:
            if mod not in selected_dates:
                continue
            output_path = os.path.join(output_dir, mod, base_name.replace('.tif', '.npy'))
            combine_and_clip_geotiff(mod_to_dir[mod], output_path, bbox_gdf, mod_to_bands[mod], year, mod, is_dated=True, selected_dates=selected_dates[mod])
        # Process static modalities (SOIL, CDL, DEM)
        for mod in ['SOIL', 'CDL', 'DEM']:
            output_path = os.path.join(output_dir, mod, base_name.replace('.tif', '.npy'))
            input_path = mod_to_dir[mod] if mod != 'DEM' else dem_path
            combine_and_clip_geotiff(input_path, output_path, bbox_gdf, mod_to_bands[mod], year, mod, is_dated=False, repeat_times=num_times)
        gc.collect()

print("Processing complete.")