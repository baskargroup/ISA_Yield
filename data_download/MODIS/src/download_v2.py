import planetary_computer
import pystac_client
import rasterio
import requests
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
from logging_helper import setup_logger
import geopandas as gpd
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

# Download function with debug logs
def download_band(asset_url, output_path):
    """Download an individual band file with a progress bar."""
    try:
        response = requests.get(asset_url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses (e.g., 403, 404)
        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as file, tqdm(
            desc=output_path, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(1024):
                file.write(chunk)
                bar.update(len(chunk))
        print(f"Downloaded: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {output_path}: {e}")

def crop_tif(input_path, output_path, bbox_lonlat):
    """Crop the GeoTIFF to the given bounding box."""
    with rasterio.open(input_path) as src:
        src_crs = src.crs
        left, bottom, right, top = transform_bounds('EPSG:4326', src_crs, *bbox_lonlat)
        window = from_bounds(left, bottom, right, top, src.transform)
        if window.width <= 0 or window.height <= 0:
            print(f"Skipping crop for {input_path}: no overlap")
            return
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile
        profile.update({
            'height': window.height,
            'width': window.width,
            'transform': transform
        })
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
        print(f"Cropped: {output_path}")

def modis_download(logger, start_date, end_date, bbox=[-96.6395, 40.3754, -90.1401, 43.5014], output_dir="./IA_modis_NBAR", unique_id=""):
    """Download and crop MODIS NBAR data for a given date range and bounding box."""
    collection = "modis-43A4-061"
    date_range = f"{start_date}/{end_date}"
    subfolder = start_date
    output_dir = os.path.join(output_dir, subfolder)
    cropped_dir = os.path.join("cropped", subfolder)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    valid_bands = {
        "Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2", "Nadir_Reflectance_Band3",
        "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band5", "Nadir_Reflectance_Band6",
        "Nadir_Reflectance_Band7"
    }

    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = pystac_client.Client.open(stac_url, modifier=planetary_computer.sign)
    search = client.search(collections=[collection], bbox=bbox, datetime=date_range)
    items = list(search.get_all_items())

    if not items:
        print(f"No MODIS NBAR images found for the date range: {start_date} to {end_date}")
        logger.info(f"No MODIS NBAR images found for the date range: {start_date} to {end_date}")
        return

    print(f"Found {len(items)} MODIS images. Downloading and cropping valid bands...\n")
    for item in items:
        scene_id = item.id
        date = item.properties['datetime'][:10]
        signed_item = planetary_computer.sign(item)
        for band_name, asset in signed_item.assets.items():
            if band_name not in valid_bands:
                continue
            band_url = asset.href
            temp_file = os.path.join(output_dir, f"{scene_id}_{band_name}.tif")
            cropped_file = os.path.join(cropped_dir, f"{band_name}_{date}_{unique_id}.tif")
            if os.path.exists(temp_file):
                print(f"⚠️ File already exists, skipping: {temp_file}")
                logger.info(f"File already exists, skipping: {temp_file}")
                continue
            if os.path.exists(cropped_file):
                print(f"⚠️ Cropped file already exists, skipping: {cropped_file}")
                continue
            try:
                download_band(band_url, temp_file)
                crop_tif(temp_file, cropped_file, bbox)
            except Exception as e:
                logger.error(f"Failed to download or crop band: {band_name} for scene: {scene_id}. Error: {e}")
                continue
    logger.info(f"Downloaded and cropped MODIS NBAR images for the date range: {start_date} to {end_date}")

def sentinel1_download(logger, start_date, end_date, bbox=[-96.6395, 40.3754, -90.1401, 43.5014], output_dir="./IA_sentinel1", unique_id=""):
    """Download Sentinel-1 RTC data for a given date range and bounding box."""
    collection = "sentinel-1-rtc"
    date_range = f"{start_date}/{end_date}"
    subfolder = start_date
    output_dir = os.path.join(output_dir, subfolder)
    cropped_dir = os.path.join("cropped", subfolder)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    valid_bands = {"vv", "vh"}

    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = pystac_client.Client.open(stac_url, modifier=planetary_computer.sign)
    search = client.search(collections=[collection], bbox=bbox, datetime=date_range)
    items = list(search.get_all_items())

    if not items:
        print(f"No Sentinel-1 images found for the date range: {start_date} to {end_date}")
        logger.info(f"No Sentinel-1 images found for the date range: {start_date} to {end_date}")
        return

    print(f"Found {len(items)} Sentinel-1 images. Downloading valid bands...\n")
    for item in items:
        scene_id = item.id
        date = item.properties['datetime'][:10]
        signed_item = planetary_computer.sign(item)
        for band_name, asset in signed_item.assets.items():
            if band_name not in valid_bands:
                continue
            band_url = asset.href
            temp_file = os.path.join(output_dir, f"{scene_id}_{band_name}.tif")
            cropped_file = os.path.join(cropped_dir, f"{band_name}_{date}_{unique_id}.tif")
            if os.path.exists(temp_file):
                print(f"⚠️ File already exists, skipping: {temp_file}")
                logger.info(f"File already exists, skipping: {temp_file}")
                continue
            if os.path.exists(cropped_file):
                print(f"⚠️ Cropped file already exists, skipping: {cropped_file}")
                continue
            try:
                download_band(band_url, temp_file)
                crop_tif(temp_file, cropped_file, bbox)
            except Exception as e:
                logger.error(f"Failed to download or crop band: {band_name} for scene: {scene_id}. Error: {e}")
                continue
    logger.info(f"Downloaded Sentinel-1 images for the date range: {start_date} to {end_date}")

def sentinel2_download(logger, start_date, end_date, bbox=[-96.6395, 40.3754, -90.1401, 43.5014], cloud_cover=10, output_dir="./IA_sentinel2", unique_id=""):
    """Download Sentinel-2 L2A data for a given date range and bounding box."""
    collection = "sentinel-2-l2a"
    date_range = f"{start_date}/{end_date}"
    subfolder = start_date
    output_dir = os.path.join(output_dir, subfolder)
    cropped_dir = os.path.join("cropped", subfolder)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    valid_bands = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}

    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = pystac_client.Client.open(stac_url, modifier=planetary_computer.sign)
    search = client.search(collections=[collection], bbox=bbox, datetime=date_range, query={"eo:cloud_cover": {"lt": cloud_cover}})
    items = list(search.get_all_items())

    if not items:
        print(f"No Sentinel-2 images found for the date range: {start_date} to {end_date}")
        logger.info(f"No Sentinel-2 images found for the date range: {start_date} to {end_date}")
        return

    print(f"Found {len(items)} Sentinel-2 images. Downloading valid bands...\n")
    for item in items:
        scene_id = item.id
        date = item.properties['datetime'][:10]
        mgrs_tile = item.properties['s2:mgrs_tile']
        signed_item = planetary_computer.sign(item)
        for band_name, asset in signed_item.assets.items():
            if band_name not in valid_bands:
                continue
            band_url = asset.href
            temp_file = os.path.join(output_dir, f"{scene_id}_{band_name}.tif")
            cropped_file = os.path.join(cropped_dir, f"{band_name}_{date}_{unique_id}_{mgrs_tile}.tif")
            if os.path.exists(temp_file):
                print(f"⚠️ File already exists, skipping: {temp_file}")
                logger.info(f"File already exists, skipping: {temp_file}")
                continue
            if os.path.exists(cropped_file):
                print(f"⚠️ Cropped file already exists, skipping: {cropped_file}")
                continue
            try:
                download_band(band_url, temp_file)
                crop_tif(temp_file, cropped_file, bbox)
            except Exception as e:
                logger.error(f"Failed to download or crop band: {band_name} for scene: {scene_id}. Error: {e}")
                continue
    logger.info(f"Downloaded Sentinel-2 images for the date range: {start_date} to {end_date}")

if __name__ == "__main__":
    # Set up the logging
    my_logger = setup_logger("./logs/satellite_download.log")

    # Read the start and end date from the CSV file
    _dates = pd.read_csv("../date_file.csv")
    # _dates = _dates[_dates["Sunday"].str.contains("2019|2020|2021|2022|2023")]
    _dates = _dates[_dates["Sunday"].str.contains("2024")]
    _dates['Sunday_dt'] = pd.to_datetime(_dates['Sunday'])
    _dates = _dates[_dates['Sunday_dt'].dt.month.between(5, 9)]
    _start_dates = _dates["Sunday"].values
    _end_dates = _dates["Saturday"].values

    # Read the Parquet file containing POINT geometries and ID column
    areas = gpd.read_parquet("../../Yield_2024.parquet")
    grouped = areas.groupby('Layer_ID')

    # Start Time
    print("\n==============================================================================================================\n")
    print(f"Downloading satellite images now at {datetime.now()}")
    print("\n==============================================================================================================\n")
    start_time = datetime.now()

    # Loop over each unique area ID
    for unique_id, group in grouped:
        xs = [point.x for point in group.geometry]
        ys = [point.y for point in group.geometry]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        area_bbox = [min_x, min_y, max_x, max_y]

        # Download images for all satellite types
        for _start_date, _end_date in zip(_start_dates, _end_dates):
            try:
                modis_download(my_logger, _start_date, _end_date, bbox=area_bbox, output_dir=f"./modis/{unique_id}", unique_id=unique_id)
                # sentinel1_download(my_logger, _start_date, _end_date, bbox=area_bbox, output_dir=f"./sentinel1/{unique_id}", unique_id=unique_id)
                # sentinel2_download(my_logger, _start_date, _end_date, bbox=area_bbox, output_dir=f"./sentinel2/{unique_id}", unique_id=unique_id)
            except Exception as e:
                my_logger.error(f"Failed to download images for area ID {unique_id} and date range: {_start_date} to {_end_date}. Error: {e}")
                continue

    # End Time
    end_time = datetime.now()
    my_logger.info(f"Downloaded satellite images in {end_time - start_time} seconds.")
    print("\n==============================================================================================================\n")
    print(f"End Time of Download: {end_time}")
    print("\n==============================================================================================================\n")
    print("\n.......................Download complete..............................\n")