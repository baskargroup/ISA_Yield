import os
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
import glob
import numpy as np
from datetime import datetime

# Define the paths
ext = "IA"
year = '2023'
modalities = ['S2L2A', 'S1GRD', 'MODIS', 'DEM', 'CDL', 'WEATHER', 'SOIL']
sentinel1_dir = f'/work/mech-ai-scratch/rtali/gis-sentinel1/final_s1_{ext}/'
sentinel2_dir = f'/work/mech-ai-scratch/rtali/gis-sentinel2/final_s2_v3_{ext}/'
modis_dir = f'/work/mech-ai-scratch/rtali/gis-modis/modis_{ext}/'
crop_dir = '/work/mech-ai-scratch/aapowadi/multimodal_fusion/remapped_cdl'
soil_dir = f'/work/mech-ai-scratch/aapowadi/soil_new/soil_processed_{ext}'
weather_dir = f'/work/mech-ai-scratch/aapowadi/WEEKLY_WEATHER_{ext}'
dem_path = f'/work/mech-ai-scratch/rtali/multimodal_fusion/terrain_merged/4326_elevation_{ext}.tif'
output_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/modalities{year}'
yield_path = '/work/mech-ai-scratch/aapowadi/ISA_Yield/modalities2023/yield_geotiffs'

# Define bands for each modality
s1_bands = ['vv', 'vh']
s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
modis_bands = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7']
weather_bands = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
dem_band = ['band_data']
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

def get_valid_dates(input_dir, bbox_gdf, band_list, year, modality):
    """Find dates with more than 70% valid data (non-NaN and non-zero)."""
    if modality == 'WEATHER':
        files = os.listdir(input_dir)
        dates = sorted(set(f.split('_')[0] for f in files if f.startswith(year)))
        dates = [d for d in dates if 4 <= int(d.split('-')[1]) <= 9]
        sorted_dates = sorted(dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    else:
        dates = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith(year + '-')]
        sorted_dates = sorted(dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))

    valid_dates = []
    for date in sorted_dates:
        band_arrays = []
        meta = None
        clip_meta = None
        if modality != 'WEATHER':
            date_dir = os.path.join(input_dir, date)
            if not all(os.path.exists(os.path.join(date_dir, f'4326_{band}.tif')) for band in band_list):
                continue
        all_bands_present = True
        for band in band_list:
            if modality != 'WEATHER':
                band_file = os.path.join(date_dir, f'4326_{band}.tif')
            else:
                band_file = os.path.join(input_dir, f'{date}_{band}.tif')
            if not os.path.exists(band_file):
                all_bands_present = False
                break
            with rasterio.open(band_file) as src:
                bbox_reproj = bbox_gdf.to_crs(src.crs)
                try:
                    out_image, out_transform = mask(src, bbox_reproj.geometry, crop=True, nodata=src.nodata)
                except ValueError:
                    all_bands_present = False
                    break
                data = out_image[0]
                if clip_meta is None:
                    clip_meta = src.meta.copy()
                    clip_meta.update({
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        'transform': out_transform
                    })
                    ref_shape = (clip_meta['height'], clip_meta['width'])
                    ref_transform = clip_meta['transform']
                    ref_crs = clip_meta['crs']
                if (out_image.shape[1], out_image.shape[2]) != ref_shape:
                    resampled_data = np.empty(ref_shape, dtype=data.dtype)
                    reproject(
                        source=data,
                        destination=resampled_data,
                        src_transform=out_transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.nearest
                    )
                    band_arrays.append(resampled_data)
                else:
                    band_arrays.append(data)
        if not all_bands_present or len(band_arrays) != len(band_list):
            continue
        stacked = np.stack(band_arrays, axis=0)
        is_finite = np.isfinite(stacked)
        is_nonzero = stacked != 0
        is_valid = is_finite & is_nonzero
        pixel_valid = np.all(is_valid, axis=0)
        fraction = np.mean(pixel_valid) if pixel_valid.size > 0 else 0
        if fraction > 0.7:
            valid_dates.append(date)
    return valid_dates

def combine_and_clip_geotiff(input_dir, output_path, bbox_gdf, band_list, year, modality, is_dated=False, selected_dates=None, target_transform=None, target_height=None, target_width=None, target_crs=None, replicate=0):
    """Combine single-band GeoTIFFs across selected dates into a multi-band GeoTIFF, clip, resample to target grid, and replicate if static."""
    try:
        band_arrays = []
        meta = None
        reference_shape = None
        reference_transform = None
        reference_crs = None

        if is_dated:
            if selected_dates is None:
                return
            sorted_dates = sorted(selected_dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
            for date in sorted_dates:
                if modality == 'WEATHER':
                    date_dir = input_dir
                else:
                    date_dir = os.path.join(input_dir, date)
                for band in band_list:
                    if modality == 'WEATHER':
                        band_file = os.path.join(date_dir, f'{date}_{band}.tif')
                    elif modality == 'SOIL':
                        band_file = os.path.join(date_dir, f'{band}.tif')
                    else:
                        band_file = os.path.join(date_dir, f'4326_{band}.tif')
                    if not os.path.exists(band_file):
                        continue
                    with rasterio.open(band_file) as src:
                        if meta is None:
                            meta = src.meta.copy()
                            reference_shape = (src.height, src.width)
                            reference_transform = src.transform
                            reference_crs = src.crs
                        data = src.read(1)
                        if (src.height, src.width) != reference_shape:
                            resampled_data = np.empty(reference_shape, dtype=data.dtype)
                            reproject(
                                source=data,
                                destination=resampled_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=reference_transform,
                                dst_crs=reference_crs,
                                resampling=Resampling.nearest
                            )
                            band_arrays.append(resampled_data)
                        else:
                            band_arrays.append(data)
        else:
            for band in band_list:
                if modality == 'SOIL':
                    band_file = os.path.join(input_dir, f'{band}.tif')
                else:
                    band_file = input_dir  # for DEM, but DEM handled separately
                if not os.path.exists(band_file):
                    print(f"Band file not found: {band_file}")
                    return
                with rasterio.open(band_file) as src:
                    if meta is None:
                        meta = src.meta.copy()
                        reference_shape = (src.height, src.width)
                        reference_transform = src.transform
                        reference_crs = src.crs
                    data = src.read(1)
                    if (src.height, src.width) != reference_shape:
                        resampled_data = np.empty(reference_shape, dtype=data.dtype)
                        reproject(
                            source=data,
                            destination=resampled_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=reference_transform,
                            dst_crs=reference_crs,
                            resampling=Resampling.nearest
                        )
                        band_arrays.append(resampled_data)
                    else:
                        band_arrays.append(data)

        if not band_arrays:
            print(f"No bands found for {input_dir}")
            return

        # Stack bands into a single array
        stacked_array = np.stack(band_arrays, axis=0)

        # Update metadata
        meta.update({
            'count': len(band_arrays)
        })

        # Reproject the bounding box to match the source CRS
        bbox_gdf_reprojected = bbox_gdf.to_crs(meta['crs'])

        # Clip the stacked image using MemoryFile
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(stacked_array)
                out_image, out_transform = mask(
                    dataset,
                    bbox_gdf_reprojected.geometry,
                    crop=True
                )

        # Update metadata for the clipped image
        meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })

        # Resample to target grid if necessary
        if target_height is not None and target_width is not None and target_transform is not None and target_crs is not None:
            if (out_image.shape[1] != target_height or out_image.shape[2] != target_width or 
                out_transform != target_transform or meta['crs'] != target_crs):
                dest_array = np.empty((out_image.shape[0], target_height, target_width), dtype=out_image.dtype)
                reproject(
                    source=out_image,
                    destination=dest_array,
                    src_transform=out_transform,
                    src_crs=meta['crs'],
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
                out_image = dest_array
                meta.update({
                    'height': target_height,
                    'width': target_width,
                    'transform': target_transform,
                    'crs': target_crs
                })

        # Replicate for static modalities
        if replicate > 0:
            num_bands = len(band_list)
            out_image = np.repeat(out_image[None, :, :, :], replicate, axis=0).reshape(replicate * num_bands, out_image.shape[1], out_image.shape[2])
            meta.update({'count': out_image.shape[0]})

        # Save the clipped multi-band image
        with rasterio.open(output_path, 'w', **meta) as dest:
            dest.write(out_image)

    except Exception as e:
        print(f"Error processing {input_dir}: {e}")

# Process each yield file
dynamic_mods = {
    'S1GRD': (sentinel1_dir, s1_bands, 'S1GRD'),
    'S2L2A': (sentinel2_dir, s2_bands, 'S2L2A'),
    'MODIS': (modis_dir, modis_bands, 'MODIS'),
    'WEATHER': (weather_dir, weather_bands, 'WEATHER')
}

for yield_file in yield_files:
    base_name = os.path.basename(yield_file)
    bbox_gdf = get_bbox_from_geotiff(yield_file)

    # Get valid dates for each dynamic modality
    valid_dates_dict = {}
    for mod, (dir_, bands, m) in dynamic_mods.items():
        valid_dates_dict[mod] = get_valid_dates(dir_, bbox_gdf, bands, year, m)

    num_dates = [len(valid_dates_dict[mod]) for mod in dynamic_mods]
    if 0 in num_dates:
        print(f"Skipping {base_name} due to modality with no valid dates.")
        continue
    min_t = min(num_dates)

    # Get target grid from S2L2A first valid date's B02 band
    target_transform = None
    target_height = None
    target_width = None
    target_crs = None
    if valid_dates_dict['S2L2A']:
        s2_valid_dates = sorted(valid_dates_dict['S2L2A'], key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        first_date = s2_valid_dates[0]
        date_dir = os.path.join(sentinel2_dir, first_date)
        band_file = os.path.join(date_dir, '4326_B02.tif')
        if os.path.exists(band_file):
            with rasterio.open(band_file) as src:
                bbox_reproj = bbox_gdf.to_crs(src.crs)
                out_image, out_transform = mask(src, bbox_reproj.geometry, crop=True)
                target_height = out_image.shape[1]
                target_width = out_image.shape[2]
                target_transform = out_transform
                target_crs = src.crs

    if target_transform is None:
        print(f"Could not determine target grid for {base_name}. Skipping.")
        continue

    # Process dynamic modalities
    for mod, (dir_, bands, m) in dynamic_mods.items():
        selected_dates = sorted(valid_dates_dict[mod], key=lambda x: datetime.strptime(x, '%Y-%m-%d'))[:min_t]
        output_path = os.path.join(output_dir, mod, base_name)
        combine_and_clip_geotiff(dir_, output_path, bbox_gdf, bands, year, m, is_dated=True, selected_dates=selected_dates,
                                 target_transform=target_transform, target_height=target_height, target_width=target_width, target_crs=target_crs)

    # Process static modalities with replication
    # CDL
    crop_path = os.path.join(crop_dir, f'{year}_{ext}_CDL.tif')
    output_path = os.path.join(output_dir, 'CDL', base_name)
    if os.path.exists(crop_path):
        with rasterio.open(crop_path) as src:
            bbox_reproj = bbox_gdf.to_crs(src.crs)
            out_image, out_transform = mask(src, bbox_reproj.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform
            })
            # Resample to target
            if (out_image.shape[1] != target_height or out_image.shape[2] != target_width or 
                out_transform != target_transform or out_meta['crs'] != target_crs):
                dest_array = np.empty((out_image.shape[0], target_height, target_width), dtype=out_image.dtype)
                reproject(
                    source=out_image,
                    destination=dest_array,
                    src_transform=out_transform,
                    src_crs=out_meta['crs'],
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
                out_image = dest_array
                out_meta.update({
                    'height': target_height,
                    'width': target_width,
                    'transform': target_transform,
                    'crs': target_crs
                })
            # Replicate
            if min_t > 0:
                out_image = np.repeat(out_image[None, :, :, :], min_t, axis=0).reshape(min_t, target_height, target_width)
                out_meta['count'] = min_t
            with rasterio.open(output_path, 'w', **out_meta) as dest:
                dest.write(out_image)

    # SOIL
    output_path = os.path.join(output_dir, 'SOIL', base_name)
    combine_and_clip_geotiff(soil_dir, output_path, bbox_gdf, soil_bands, year, 'SOIL', is_dated=False, selected_dates=None,
                             target_transform=target_transform, target_height=target_height, target_width=target_width, target_crs=target_crs, replicate=min_t)

    # DEM
    output_path = os.path.join(output_dir, 'DEM', base_name)
    if os.path.exists(dem_path):
        with rasterio.open(dem_path) as src:
            bbox_reproj = bbox_gdf.to_crs(src.crs)
            out_image, out_transform = mask(src, bbox_reproj.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform
            })
            # Resample to target
            if (out_image.shape[1] != target_height or out_image.shape[2] != target_width or 
                out_transform != target_transform or out_meta['crs'] != target_crs):
                dest_array = np.empty((out_image.shape[0], target_height, target_width), dtype=out_image.dtype)
                reproject(
                    source=out_image,
                    destination=dest_array,
                    src_transform=out_transform,
                    src_crs=out_meta['crs'],
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
                out_image = dest_array
                out_meta.update({
                    'height': target_height,
                    'width': target_width,
                    'transform': target_transform,
                    'crs': target_crs
                })
            # Replicate
            if min_t > 0:
                out_image = np.repeat(out_image[None, :, :, :], min_t, axis=0).reshape(min_t, target_height, target_width)
                out_meta['count'] = min_t
            with rasterio.open(output_path, 'w', **out_meta) as dest:
                dest.write(out_image)

print("Processing complete.")