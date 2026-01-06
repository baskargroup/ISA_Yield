import pandas as pd
import numpy as np
from math import ceil
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import geopandas as gpd
import os
import glob
from pyproj import Transformer


def create_geotiff_from_group(df, res_m=10.0, utm_epsg="EPSG:32615"):
    groups = df.groupby("Layer_ID")

    for layer_id, group in groups:
        if group.empty:
            continue

        # --- 1) Reproject lon/lat -> UTM (meters)
        transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

        lon = group["x"].to_numpy()
        lat = group["y"].to_numpy()

        x_utm, y_utm = transformer.transform(lon, lat)
        points = np.column_stack([x_utm, y_utm])
        values = group["Yield"].to_numpy()

        # --- 2) Snap bounds to a 10 m grid so pixels are EXACTLY res_m
        min_x, max_x = float(np.min(x_utm)), float(np.max(x_utm))
        min_y, max_y = float(np.min(y_utm)), float(np.max(y_utm))

        # Snap outward to multiples of res_m
        left = np.floor(min_x / res_m) * res_m
        right = np.ceil(max_x / res_m) * res_m
        bottom = np.floor(min_y / res_m) * res_m
        top = np.ceil(max_y / res_m) * res_m

        # --- 3) Compute exact raster shape from snapped bounds
        ncols = int(round((right - left) / res_m))
        nrows = int(round((top - bottom) / res_m))

        if ncols <= 0 or nrows <= 0:
            continue

        # --- 4) Build transform with from_origin (top-left corner) and exact res
        # from_origin(west, north, xsize, ysize)
        transform = from_origin(left, top, res_m, res_m)

        # --- 5) Build grid of CELL CENTERS (no linspace ambiguity)
        # Cell center coordinates:
        # x_center = left + (col + 0.5) * res_m
        # y_center = top  - (row + 0.5) * res_m
        x_centers = left + (np.arange(ncols) + 0.5) * res_m
        y_centers = top - (np.arange(nrows) + 0.5) * res_m
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)

        # --- 6) Interpolate onto center grid
        interpolated = griddata(points, values, (x_grid, y_grid), method="linear")

        # --- 7) Distance mask in meters (exact)
        tree = cKDTree(points)
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        max_distance_m = 3 * res_m  # 30 m (3 pixels)
        distances, _ = tree.query(grid_points)
        distances = distances.reshape((nrows, ncols))

        data = np.where(distances <= max_distance_m, interpolated, np.nan)

        # Optional: keep nodata as NaN and write a nodata value (recommended)
        nodata_value = -9999.0
        data_out = np.where(np.isnan(data), nodata_value, data).astype("float32")

        # --- 8) Save GeoTIFF
        os.makedirs("unprocessed_data/yield_geotiffs", exist_ok=True)
        if len(group.Crop.unique()) == 1:
            output_file = f"unprocessed_data/yield_geotiffs/{layer_id}_{group['Crop'].iloc[0]}.tif"
        else:
            output_file = f"unprocessed_data/yield_geotiffs/{layer_id}_mixed.tif"

        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=nrows,
            width=ncols,
            count=1,
            dtype="float32",
            crs=utm_epsg,
            transform=transform,
            nodata=nodata_value,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(data_out, 1)

        print(f"Created GeoTIFF (UTM): {output_file}")
        print(f"  - Pixel size: {res_m}m × {res_m}m (exact)")
        print(f"  - Grid: {ncols} cols × {nrows} rows")
        print(f"  - CRS: {utm_epsg}")
        print(f"  - Nodata: {nodata_value}")


# Process all available years
parquet_files = sorted(glob.glob("Yield_*_filtered.parquet"))

if not parquet_files:
    print("No filtered yield parquet files found.")
else:
    for parquet_file in parquet_files:
        year = parquet_file.split("_")[1]
        print(f"\nProcessing year: {year}")
        full_data = gpd.read_parquet(parquet_file)
        create_geotiff_from_group(full_data, res_m=10.0, utm_epsg="EPSG:32615")

print("\n✅ Processing complete. All GeoTIFFs saved with exact 10m pixels and cell-center grids.")
