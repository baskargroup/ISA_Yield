import planetary_computer
import pystac_client
import rasterio
import requests
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
from logging_helper import setup_logger


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


def modis_download(logger, start_date, end_date, bbox=[-96.6395, 40.3754, -90.1401, 43.5014], output_dir="./IA_modis_NBAR"):

    # Define parameters
    collection = "modis-43A4-061"  # Landsat 9 Collection 2 Level 2
    bbox = bbox  # Bounding box for Iowa (West, South, East, North)
    date_range = start_date+"/"+end_date  # Example 1-week range

    subfolder = start_date

    # Create subfolder under output_dir
    output_dir = os.path.join(output_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    # Valid Landsat 9 bands (excluding angle & metadata bands)
    valid_bands = {
        "Nadir_Reflectance_Band1", "Nadir_Reflectance_Band2", "Nadir_Reflectance_Band3", "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band5", "Nadir_Reflectance_Band6", "Nadir_Reflectance_Band7",
    }

    # Connect to the Microsoft Planetary Computer STAC API
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = pystac_client.Client.open(
        stac_url, modifier=planetary_computer.sign)

    # Search for Landsat 9 images
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,

    )
    items = list(search.get_all_items())

    if not items:
        print("No MODIS NBAR images found for the given criteria.")
        logger.info(
            f"No MODIS NBAR images found for the date range: {start_date} to {end_date}")

    else:
        print(f"Found {len(items)} images. Downloading valid bands...\n")

        # Process each Landsat image
        for item in items:
            scene_id = item.id
            # print(f"\n🔍 Processing Landsat scene: {scene_id}")

            # Ensure STAC item has valid assets
            signed_item = planetary_computer.sign(item)

            # print("🔍 Searching for valid bands...")
            # print(f"🔍 Band assets: {list(signed_item.assets.keys())}" )

            for band_name, asset in signed_item.assets.items():
                if band_name not in valid_bands:
                    continue  # Skip invalid bands

                band_url = asset.href  # Get signed URL
                output_file = os.path.join(
                    output_dir, f"{scene_id}_{band_name}.tif")

                # Check if the file already exists
                if os.path.exists(output_file):
                    print(f"⚠️ File already exists, skipping: {output_file}")
                    logger.info(
                        f"File already exists, skipping: {output_file}")
                    continue

                # print(f"⬇️ Downloading {band_name} from {band_url} ...")
                # print(f"Band URL: {band_url}")
                try:
                    download_band(band_url, output_file)
                except Exception as e:
                    logger.error(
                        f"Failed to download band: {band_name} for the scene: {scene_id}. Error: {e}")
                    continue

                # Try to open and get metadata
                # try:
                #     with rasterio.open(output_file) as dataset:
                #         crs = dataset.crs  # Coordinate Reference System
                #         width, height = dataset.width, dataset.height  # Image size
                #         print(f"📏 Band: {band_name}, CRS: {crs}, Size: {width}x{height} pixels")
                # except rasterio.errors.RasterioIOError:
                #     print(f"⚠️ Skipping non-image band: {band_name}")

        logger.info(
            f"Downloaded MODIS NBAR images for the date range: {start_date} to {end_date}")


if __name__ == "__main__":

    # Set up the logging
    os.makedirs('./logs',exist_ok=True)
    my_logger = setup_logger("./logs/modis_NBAR_download.log")

    # Read the start and end date from the CSV file
    _dates = pd.read_csv("../date_file.csv")
    # _dates = _dates[_dates["Sunday"].str.contains("2019|2020|2021|2022|2023")]
    _dates = _dates[_dates["Sunday"].str.contains("2024")]
    _dates['Sunday_dt'] = pd.to_datetime(_dates['Sunday'])
    _dates = _dates[_dates['Sunday_dt'].dt.month.between(5, 9)]
    _start_dates = _dates["Sunday"].values
    _end_dates = _dates["Saturday"].values

    # print(f"Number of dates: {len(_start_dates)}")

    # Start Time
    print("\n==============================================================================================================\n")
    print(f"Downloading MODIS NBAR images now at {datetime.now()}")
    print("\n==============================================================================================================\n")
    start_time = datetime.now()

    # Download the Landsat 9 images for the start and end dates
    for _start_date, _end_date in zip(_start_dates, _end_dates):
        try:
            modis_download(my_logger, _start_date, _end_date)
        except Exception as e:
            my_logger.error(
                f"Failed to download modis SURF images for the date range: {_start_date} to {_end_date}. Error: {e}")
            continue

    # End Time
    end_time = datetime.now()
    my_logger.info(
        f"Downloaded MODIS NBAR images for the date range: {_start_date} to {_end_date} in {end_time - start_time} seconds.")

    print("\n==============================================================================================================\n")
    print(f"End Time of Download: {end_time}")
    print("\n==============================================================================================================\n")

    print("\n.......................Download complete..............................\n")
