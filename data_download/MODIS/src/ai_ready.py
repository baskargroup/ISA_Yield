import os
import time
import glob
import pandas as pd
import xarray as xr
import rioxarray
import warnings
warnings.filterwarnings("ignore")

def stack_by_band(global_pattern):
    
    _folders = glob.glob(global_pattern)
    _folders = sorted(_folders)
    
    for _folder in _folders:
        
        stack_time =  time.time()
        
        print(f"Processing folder: {_folder}")
    
        tif_files = {
        "Band1": f"{_folder}4326_Band1.tif",
        "Band2": f"{_folder}4326_Band2.tif",
        "Band3": f"{_folder}4326_Band3.tif",
        "Band4": f"{_folder}4326_Band4.tif",
        "Band5": f"{_folder}4326_Band5.tif",
        "Band6": f"{_folder}4326_Band6.tif",
        "Band7": f"{_folder}4326_Band7.tif",
        }

        # Create an empty list to store datasets
        datasets = []

        # Open each GeoTIFF file and add it to the list
        for var_name, file_path in tif_files.items():
            # Open the GeoTIFF file using rioxarray
            da = rioxarray.open_rasterio(file_path)
            
            # Convert DataArray to Dataset with a meaningful variable name
            ds = da.to_dataset(name=var_name)
            
            # If the dataset has a 'band' dimension with only one band, we can squeeze it
            # if 'band' in ds.dims and ds.dims['band'] == 1:
            #     ds = ds.squeeze('band')
            
            datasets.append(ds)

        # Merge all datasets into a single dataset
        merged_ds = xr.merge(datasets)
        
        output_path = os.path.join(_folder, "modis.nc")

        # Save the merged dataset as a NetCDF file
        try:
            merged_ds.to_netcdf(output_path, engine='netcdf4')
        except Exception as e:
            print(f"Error saving file: {e}")
            
        print(f"Stacking time for folder {_folder}: {time.time() - stack_time}")

    print("Conversion complete. All GeoTIFF files have been merged into 'modis.nc'.")


def aggregate_date_centric(base_path_pattern, year, variables):
    """Aggregate GeoTIFFs organized by date folders with variable-based filenames."""
    # Get all date folders
    date_folders = sorted(glob.glob(base_path_pattern))
    date_names = [os.path.basename(os.path.dirname(f)) for f in date_folders]
    dates = pd.to_datetime(date_names, format='ISO8601')
    
    # Create list to hold data for each date
    date_datasets = []
    
    for i, date_folder in enumerate(date_folders):
        # Create dict to hold all variables for this date
        var_arrays = {}
        
        for var_name in variables:
            # Path to the specific variable file for this date
            file_path = os.path.join(date_folder, f"4326_{var_name}.tif")
            
            if os.path.exists(file_path):
                # Open the raster
                da = rioxarray.open_rasterio(file_path)
                
                # If there's a band dimension with only one band, squeeze it
                if 'band' in da.dims and da.sizes['band'] == 1:
                    da = da.squeeze('band')
                
                var_arrays[var_name] = da
        
        # Create a dataset for this date with all variables
        if var_arrays:
            date_ds = xr.Dataset(var_arrays)
            date_ds = date_ds.expand_dims(dim={"time": [dates[i]]})
            date_datasets.append(date_ds)
    
    # Concatenate all date datasets along the time dimension
    if date_datasets:
        combined_ds = xr.concat(date_datasets, dim="time")
        return combined_ds
    else:
        raise ValueError("No valid data found")
    

if __name__ == "__main__":
    year = 2020
    base_path_pattern = f"./modis/{year}-*/"
    bands = ["Band1", "Band2", "Band3", "Band4", "Band5", "Band6", "Band7"]
    
    stack_by_band(base_path_pattern)
    
    # combined_ds = aggregate_date_centric(base_path_pattern, year, bands)
    # combined_ds.to_netcdf(f"./modis/modis_{year}.nc", engine="netcdf4")