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
output_dir = f'/work/mech-ai-scratch/aapowadi/ISA_Yield/modalities{year}'
yield_path = '/work/mech-ai-scratch/aapowadi/ISA_Yield/unprocessed_data/yield_geotiffs'

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
        crs = src.crs
    return gpd.GeoDataFrame({'geometry': [bbox]}, crs=crs)

def load_band(band_file, bbox_gdf, ref_shape=None, ref_transform=None, ref_crs=None):
    """Load a single band using geospatial windowed reading."""
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

def combine_and_clip_geotiff(input_dir, output_path, bbox_gdf, band_list, year, modality, is_dated=False):
    """Combine single-band GeoTIFFs into a multi-band GeoTIFF and clip to the bounding box using windowed reading."""
    try:
        if is_dated:
            dates = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
            dates = [d for d in dates if d.startswith(year + '-')]
            sorted_dates = sorted(dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
            selected_dir = None
            for date in sorted_dates:
                date_dir = os.path.join(input_dir, date)
                if all(os.path.exists(os.path.join(date_dir, f'4326_{band}.tif')) for band in band_list):
                    selected_dir = date_dir
                    break
            if selected_dir is None:
                print(f"No date found with all bands available for {input_dir}")
                return
        else:
            selected_dir = input_dir
            if modality == 'WEATHER':
                # Extract unique dates from weather files by removing band suffix
                files = os.listdir(input_dir)
                dates = sorted(set(f.split('_')[0] for f in files if f.startswith(year)))
                dates = [d for d in dates if 4 <= int(d.split('-')[1]) <= 9]
                if not dates:
                    print(f"No valid weather dates found for {input_dir}")
                    return
                sorted_dates = sorted(dates, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
                selected_date = sorted_dates[0]

        band_arrays = []
        meta = None
        reference_shape = None
        reference_transform = None
        reference_crs = None

        for band in band_list:
            if modality not in ['WEATHER', 'SOIL']:
                band_file = os.path.join(selected_dir, f'4326_{band}.tif')
            elif modality == 'WEATHER':
                band_file = os.path.join(input_dir, f'{selected_date}_{band}.tif')
            elif modality == 'SOIL':
                band_file = os.path.join(input_dir, f'{band}.tif')
            if not os.path.exists(band_file):
                print(f"Band file not found: {band_file}")
                return
            if reference_shape is None:
                data, reference_shape, reference_transform, reference_crs, meta = load_band(band_file, bbox_gdf)
                band_arrays.append(data)
            else:
                data = load_band(band_file, bbox_gdf, reference_shape, reference_transform, reference_crs)
                band_arrays.append(data)
            del data
            gc.collect()

        if not band_arrays:
            print(f"No bands found for {input_dir}")
            return

        # Stack bands into a single array
        stacked_array = np.stack(band_arrays, axis=0)
        meta.update({
            'count': len(band_arrays),
            'height': reference_shape[0],
            'width': reference_shape[1],
            'transform': reference_transform
        })

        # Reproject the bounding box to match the source CRS
        bbox_gdf_reprojected = bbox_gdf.to_crs(reference_crs)

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

        # Save the clipped multi-band image
        with rasterio.open(output_path, 'w', **meta) as dest:
            dest.write(out_image)

        del stacked_array, out_image, band_arrays
        gc.collect()

    except Exception as e:
        print(f"Error processing {input_dir}: {e}")

# Process each yield file
for yield_file in yield_files:
    base_name = os.path.basename(yield_file)
    bbox_gdf = get_bbox_from_geotiff(yield_file)

    # Process Sentinel-1 images
    output_path = os.path.join(output_dir, 'S1GRD', base_name)
    combine_and_clip_geotiff(sentinel1_dir, output_path, bbox_gdf, s1_bands, year, 'S1GRD', is_dated=True)

    # Process Sentinel-2 images
    output_path = os.path.join(output_dir, 'S2L2A', base_name)
    combine_and_clip_geotiff(sentinel2_dir, output_path, bbox_gdf, s2_bands, year, 'S2L2A', is_dated=True)

    # Process MODIS images
    output_path = os.path.join(output_dir, 'MODIS', base_name)
    combine_and_clip_geotiff(modis_dir, output_path, bbox_gdf, modis_bands, year, 'MODIS', is_dated=True)

    # Process crop images
    crop_path = os.path.join(crop_dir, f'{year}_{ext}_CDL.tif')
    output_path = os.path.join(output_dir, 'CDL', base_name)
    with rasterio.open(crop_path) as src:
        bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
        bounds = bbox_gdf_reprojected.geometry.iloc[0].bounds
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            print(f"Invalid window for {crop_path}: {window}")
            continue
        out_image = src.read(window=window)
        out_transform = src.window_transform(window)
        out_meta = src.meta.copy()
        out_meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(out_image)
        del out_image
        gc.collect()

    # Process soil images
    output_path = os.path.join(output_dir, 'SOIL', base_name)
    combine_and_clip_geotiff(soil_dir, output_path, bbox_gdf, soil_bands, year, "SOIL", is_dated=False)

    # Process weather images
    output_path = os.path.join(output_dir, 'WEATHER', base_name)
    combine_and_clip_geotiff(weather_dir, output_path, bbox_gdf, weather_bands, year, 'WEATHER', is_dated=False)

    # Process DEM image
    output_path = os.path.join(output_dir, 'DEM', base_name)
    with rasterio.open(dem_path) as src:
        bbox_gdf_reprojected = bbox_gdf.to_crs(src.crs)
        bounds = bbox_gdf_reprojected.geometry.iloc[0].bounds
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            print(f"Invalid window for {dem_path}: {window}")
            continue
        out_image = src.read(window=window)
        out_transform = src.window_transform(window)
        out_meta = src.meta.copy()
        out_meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(out_image)
        del out_image
        gc.collect()

print("Processing complete.")