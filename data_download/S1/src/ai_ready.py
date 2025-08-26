import os
import glob
import pandas as pd
import xarray as xr
import rioxarray

def aggregate_date_centric(base_path_pattern, year, variables):
    """Aggregate GeoTIFFs organized by date folders with variable-based filenames."""
    # Get all date folders
    date_folders = sorted(glob.glob(base_path_pattern))
    date_names = [os.path.basename(os.path.dirname(f)) for f in date_folders]
    dates = pd.to_datetime(date_names, format='%Y%m%d')
    
    # Create list to hold data for each date
    date_datasets = []
    
    for i, date_folder in enumerate(date_folders):
        # Create dict to hold all variables for this date
        var_arrays = {}
        
        for var_name in variables:
            # Path to the specific variable file for this date
            file_path = os.path.join(date_folder, f"{var_name}.tif")
            
            if os.path.exists(file_path):
                # Open the raster
                da = rioxarray.open_rasterio(file_path)
                
                # If there's a band dimension with only one band, squeeze it
                if 'band' in da.dims and da.dims['band'] == 1:
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
    
    # Define the base path and variables to aggregate
    base_path_pattern = f"./final_s1/{year}-*"
    bands = ["vv", "vh"]
    
    # Aggregate the data
    combined_ds = aggregate_date_centric(base_path_pattern=base_path_pattern, year = year, variables=bands)
    combined_ds.to_netcdf(f"./final_s1/sentinel1_{year}.nc", engine="netcdf4")