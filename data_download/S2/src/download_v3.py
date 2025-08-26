import planetary_computer
import pystac_client
import rasterio
import requests
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
from logging_helper import setup_logger
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download function with debug logs


def download_band(asset_url, output_path):
    """Download an individual band file with a progress bar."""
    try:
        response = requests.get(asset_url, stream=True)
        response.raise_for_status()
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


def sentinel2_download(logger, start_date, end_date, bbox=None, cloud_cover=None, output_dir=None):
    """Download Sentinel-2 images for a given date range."""
    bbox = bbox or [-96.6395, 40.3754, -90.1401, 43.5014]  # Default Iowa bbox
    cloud_cover = cloud_cover or 10  # Default cloud cover
    output_dir = output_dir or "./IA_sentinel2"  # Default output directory

    # Define parameters
    collection = "sentinel-2-l2a"
    date_range = f"{start_date}/{end_date}"
    subfolder = start_date
    output_dir = os.path.join(output_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    valid_bands = {"B01", "B02", "B03", "B04", "B05",
                   "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}

    # Connect to the Microsoft Planetary Computer STAC API
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = pystac_client.Client.open(
        stac_url, modifier=planetary_computer.sign)

    # Search for Sentinel-2 images
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover}},
    )
    items = list(search.get_all_items())

    if not items:
        print(f"No Sentinel-2 images found for {start_date} to {end_date}.")
        logger.info(
            f"No Sentinel-2 images found for the date range: {start_date} to {end_date}")
    else:
        print(
            f"Found {len(items)} images for {start_date} to {end_date}. Downloading valid bands...")

        for item in items:
            scene_id = item.id
            signed_item = planetary_computer.sign(item)

            for band_name, asset in signed_item.assets.items():
                if band_name not in valid_bands:
                    continue

                band_url = asset.href
                output_file = os.path.join(
                    output_dir, f"{scene_id}_{band_name}.tif")

                if os.path.exists(output_file):
                    print(f"⚠️ File already exists, skipping: {output_file}")
                    logger.info(
                        f"File already exists, skipping: {output_file}")
                    continue

                try:
                    download_band(band_url, output_file)
                except Exception as e:
                    logger.error(
                        f"Failed to download band: {band_name} for scene: {scene_id}. Error: {e}")
                    continue

        logger.info(
            f"Downloaded Sentinel-2 images for the date range: {start_date} to {end_date}")


if __name__ == "__main__":
    # Set up the logging
    my_logger = setup_logger("./logs/sentinel2_download.log")

    # Read the start and end dates from the CSV file
    _dates = pd.read_csv("sundays_and_saturdays_2004_2024.csv")
    _dates = _dates[_dates["Sunday"].str.contains("2019")]
    _dates = _dates[_dates["Sunday"] >= "2019-08-18"]
    _start_dates = _dates["Sunday"].values
    _end_dates = _dates["Saturday"].values

    print(f"Number of date ranges to process: {len(_start_dates)}")

    # Start time
    print("\n==============================================================================================================\n")
    print(f"Downloading Sentinel-2 images now at {datetime.now()}")
    print("\n==============================================================================================================\n")
    start_time = datetime.now()

    # Use ThreadPoolExecutor with 8 workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks to the executor
        future_to_date = {
            executor.submit(sentinel2_download, my_logger, _start_date, _end_date): (_start_date, _end_date)
            for _start_date, _end_date in zip(_start_dates, _end_dates)
        }

        # Process results as they complete
        for future in as_completed(future_to_date):
            start_date, end_date = future_to_date[future]
            try:
                future.result()  # This will raise any exceptions that occurred during execution
            except Exception as e:
                my_logger.error(
                    f"Failed to download Sentinel-2 images for {start_date} to {end_date}. Error: {e}")

    # End time
    end_time = datetime.now()
    my_logger.info(
        f"Downloaded Sentinel-2 images for all date ranges in {end_time - start_time} seconds.")

    print("\n==============================================================================================================\n")
    print(f"End Time of Download: {end_time}")
    print("\n==============================================================================================================\n")
    print("\n.......................Download complete..............................\n")
