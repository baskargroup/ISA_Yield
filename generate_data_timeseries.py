import os
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
import glob
import numpy as np
from datetime import datetime
import pdb
# Define the paths
ext = "IA"
year = '2024'
modalities = ['S2L2A', 'S1GRD', 'MODIS', 'DEM', 'CDL', 'WEATHER', 'SOIL']
sentinel1_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/data_download/S1/final_s1_{ext}/'
sentinel2_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/data_download/S2/final_s2_v3_{ext}/'
modis_dir = f'/work/mech-ai-scratch/rtali/gis-modis/modis_{ext}/'
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

# Ensure output directories exist
for m in modalities:
    os.makedirs(os.path.join(output_dir, m), exist_ok=True)

# Get list of yield geotiff files
yield_files = glob.glob(os.path.join(yield_path, '*.tif'))

def get_bbox_from_geotiff(file_path):
    """Extract the bounding box from a geotiff file."""
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    return gpd.GeoDataFrame({'geometry': [bbox]}, crs=src.crs)

def load_band(band_file, ref_shape=None, ref_transform=None, ref_crs=None):
    with rasterio.open(band_file) as src:
        data = src.read(1)
        if ref_shape is None:
            return data, src.shape, src.transform, src.crs, src.meta.copy()
        else:
            resampled_data = np.empty(ref_shape, dtype=data.dtype)
            reproject(
                source=data,
                destination=resampled_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest
            )
            return resampled_data

def get_stacked_for_date(input_dir, bbox_gdf, band_list, date, modality):
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
                data, ref_shape, ref_transform, ref_crs, meta = load_band(band_file)
                band_arrays.append(data)
            else:
                data = load_band(band_file, ref_shape, ref_transform, ref_crs)
                band_arrays.append(data)
        if len(band_arrays) < len(band_list):
            return None
        stacked_array = np.stack(band_arrays, axis=0)
        meta.update({'count': stacked_array.shape[0]})
        bbox_gdf_reprojected = bbox_gdf.to_crs(meta['crs'])
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(stacked_array)
                out_image, out_transform = mask(
                    dataset,
                    bbox_gdf_reprojected.geometry,
                    crop=True
                )
        return out_image
    except Exception as e:
        print(f"Error getting stacked for date {date} in {modality}: {e}")
        return None

def is_valid_date(input_dir, bbox_gdf, band_list, date, modality):
    out_image = get_stacked_for_date(input_dir, bbox_gdf, band_list, date, modality)
    if out_image is None:
        return False
    if modality != 'WEATHER':
        valid_per_band = ~np.isnan(out_image) & (out_image != 0)
    else:
        valid_per_band = ~np.isnan(out_image)
    all_valid = np.all(valid_per_band, axis=0)
    frac_valid = np.mean(all_valid)
    return frac_valid > 0.7

def combine_and_clip_geotiff(input_dir, output_path, bbox_gdf, band_list, year, modality, is_dated=False, selected_dates=None, repeat_times=1):
    try:
        if selected_dates is None:
            selected_dates = []
        ref_shape = None
        ref_transform = None
        ref_crs = None
        meta = None
        band_arrays = []
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
                        data, ref_shape, ref_transform, ref_crs, meta = load_band(band_file)
                        band_arrays.append(data)
                    else:
                        data = load_band(band_file, ref_shape, ref_transform, ref_crs)
                        band_arrays.append(data)
        else:
            for band in band_list:
                if modality == 'SOIL':
                    band_file = os.path.join(input_dir, f'{band}.tif')
                if not os.path.exists(band_file):
                    continue
                if ref_shape is None:
                    data, ref_shape, ref_transform, ref_crs, meta = load_band(band_file)
                    band_arrays.append(data)
                else:
                    data = load_band(band_file, ref_shape, ref_transform, ref_crs)
                    band_arrays.append(data)
        if not band_arrays:
            print(f"No bands found for {input_dir}")
            return
        stacked_array = np.stack(band_arrays, axis=0)
        if not is_dated and repeat_times > 1:
            stacked_array = np.tile(stacked_array, (repeat_times, 1, 1))
        meta.update({
            'count': stacked_array.shape[0]
        })
        bbox_gdf_reprojected = bbox_gdf.to_crs(meta['crs'])
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(stacked_array)
                out_image, out_transform = mask(
                    dataset,
                    bbox_gdf_reprojected.geometry,
                    crop=True
                )
        meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        height = out_image.shape[1]
        width = out_image.shape[2]
        out_arr = np.reshape(out_image, (-1,len(band_list),height,width))
        np.save(output_path, out_arr)
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")

# Define dynamic and static modalities
dynamic_mods = ['S1GRD', 'S2L2A', 'MODIS', 'WEATHER']
mod_to_dir = {
    'S1GRD': sentinel1_dir,
    'S2L2A': sentinel2_dir,
    'MODIS': modis_dir,
    'WEATHER': weather_dir,
    'SOIL': soil_dir
}
mod_to_bands = {
    'S1GRD': s1_bands,
    'S2L2A': s2_bands,
    'MODIS': modis_bands,
    'WEATHER': weather_bands,
    'SOIL': soil_bands
}

# Process each yield file
for yield_file in yield_files:
    base_name = os.path.basename(yield_file)
    bbox_gdf = get_bbox_from_geotiff(yield_file)
    valid_dates = {}
    for mod in dynamic_mods:
        input_dir = mod_to_dir[mod]
        band_list = mod_to_bands[mod]
        if mod == 'WEATHER':
            files = os.listdir(input_dir)
            possible_dates = sorted(set(f.split('_')[0] for f in files if f.startswith(year)))
            possible_dates = [d for d in possible_dates if 4 <= int(d.split('-')[1]) <= 9]
        else:
            possible_dates = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith(year + '-')]
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
    min_len = min(len(valid_dates[mod]) for mod in dynamic_mods if len(valid_dates[mod]) > 0)
    ref_mod = min((m for m in dynamic_mods if len(valid_dates[m]) > 0), key=lambda m: len(valid_dates[m]))
    ref_dates = sorted(valid_dates[ref_mod])
    num_times = len(ref_dates)
    selected_dates = {}
    for mod in dynamic_mods:
        if len(valid_dates[mod]) == 0:
            print(f"No valid dates for {mod} in {base_name}, skipping modality.")
            continue
        mod_val_dates = sorted(valid_dates[mod])
        mod_val_dt = [datetime.strptime(d, '%Y-%m-%d') for d in mod_val_dates]
        sel = []
        for rd in ref_dates:
            rd_dt = datetime.strptime(rd, '%Y-%m-%d')
            i_closest = np.argmin([abs(dt - rd_dt) for dt in mod_val_dt])
            sel.append(mod_val_dates[i_closest])
        selected_dates[mod] = sel
    # Process dynamic modalities
    for mod in dynamic_mods:
        if mod not in selected_dates:
            continue
        output_path = os.path.join(output_dir, mod, base_name)
        combine_and_clip_geotiff(mod_to_dir[mod], output_path, bbox_gdf, mod_to_bands[mod], year, mod, is_dated=True, selected_dates=selected_dates[mod])
    # Process SOIL
    output_path = os.path.join(output_dir, 'SOIL', base_name)
    combine_and_clip_geotiff(soil_dir, output_path, bbox_gdf, soil_bands, year, 'SOIL', is_dated=False, repeat_times=num_times)
    # Process CDL
    crop_path = os.path.join(crop_dir, f'{year}_{ext}_CDL.tif')
    output_path = os.path.join(output_dir, 'CDL', base_name)
    with rasterio.open(crop_path) as src:
        bbox_rep = bbox_gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, bbox_rep.geometry, crop=True)
        out_meta = src.meta.copy()
        if num_times > 1:
            out_image = np.tile(out_image, (num_times, 1, 1))
            out_meta['count'] = num_times * out_meta.get('count', 1)
        out_meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        height = out_image.shape[1]
        width = out_image.shape[2]
        out_arr = np.reshape(out_image, (-1,1,height,width))
        np.save(output_path,out_arr)
    # Process DEM
    output_path = os.path.join(output_dir, 'DEM', base_name)
    with rasterio.open(dem_path) as src:
        bbox_rep = bbox_gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, bbox_rep.geometry, crop=True)
        out_meta = src.meta.copy()
        if num_times > 1:
            out_image = np.tile(out_image, (num_times, 1, 1))
            out_meta['count'] = num_times * out_meta.get('count', 1)
        out_meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        height = out_image.shape[1]
        width = out_image.shape[2]
        out_arr = np.reshape(out_image, (-1,1,height,width))
        np.save(output_path,out_arr)

print("Processing complete.")