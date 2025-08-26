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

# Optimized download function
def download_band(asset_url, output_path, chunk_size=8192):
    """Download an individual band file with a progress bar."""
    try:
        response = requests.get(asset_url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as file, tqdm(
            desc=os.path.basename(output_path), total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size):
                file.write(chunk)
                bar.update(len(chunk))

        return f"Downloaded: {output_path}"
    except requests.exceptions.RequestException as e:
        return f"Failed to download {output_path}: {e}"

def sentinel1_download(logger, start_date, end_date, bbox=[-96.6395, 40.3754, -90.1401, 43.5014], output_dir="./IA_sentinel1", max_workers=8):
    # Define parameters
    collection = "sentinel-1-rtc"
    date_range = f"{start_date}/{end_date}"
    subfolder = start_date
    output_dir = os.path.join(output_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    valid_bands = {"vv", "vh"}

    # Connect to STAC API
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = pystac_client.Client.open(stac_url, modifier=planetary_computer.sign)

    # Search for Sentinel-1 images
    search = client.search(collections=[collection], bbox=bbox, datetime=date_range)
    items = list(search.get_all_items())

    if not items:
        print(f"No Sentinel-1 images found for {date_range}.")
        logger.info(f"No Sentinel-1 images found for {date_range}")
        return

    print(f"Found {len(items)} images for {date_range}. Downloading valid bands...")

    # Prepare download tasks
    download_tasks = []
    for item in items:
        scene_id = item.id
        signed_item = planetary_computer.sign(item)
        for band_name, asset in signed_item.assets.items():
            if band_name not in valid_bands:
                continue
            band_url = asset.href
            output_file = os.path.join(output_dir, f"{scene_id}_{band_name}.tif")
            if not os.path.exists(output_file):  # Optional: skip if file exists
                download_tasks.append((band_url, output_file))

    # Parallel downloads using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(download_band, url, path): path for url, path in download_tasks}
        for future in as_completed(future_to_file):
            result = future.result()
            print(result)
            logger.info(result)

    logger.info(f"Completed downloads for {date_range}")

def main():
    # Set up logging
    my_logger = setup_logger("./logs/sentinel1_download.log")

    # Read dates
    _dates = pd.read_csv("date_file.csv")
    _dates = _dates[_dates["Sunday"].str.contains("2019")]
    start_dates = _dates["Sunday"].values
    end_dates = _dates["Saturday"].values

    print(f"\nStarting download at {datetime.now()}\n{'='*50}")
    start_time = datetime.now()

    # Parallelize across date ranges (optional, uncomment to use)
    # from multiprocessing import Pool
    # with Pool(processes=2) as pool:
    #     pool.starmap(sentinel1_download, [(my_logger, s, e) for s, e in zip(start_dates, end_dates)])

    # Sequential execution (default)
    for start_date, end_date in zip(start_dates, end_dates):
        try:
            sentinel1_download(my_logger, start_date, end_date, max_workers=4)
        except Exception as e:
            my_logger.error(f"Failed for {start_date} to {end_date}: {e}")

    end_time = datetime.now()
    my_logger.info(f"Total time: {end_time - start_time}")
    print(f"\nEnd time: {end_time}\n{'='*50}\nDownload complete.")

if __name__ == "__main__":
    main()