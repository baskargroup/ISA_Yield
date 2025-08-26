import glob
import pandas as pd
import numpy as np

def check_date(file_name, dt):
    _date = file_name.split("_")[-6]
    _date = _date.split("T")[0]
    
    print(f"Date: {_date}")
    
    #Convert to datetime. _date is a string in the format YYYYMMDD
    _date = pd.to_datetime(_date, format="%Y%m%d")
    
    #Convert dt to datetime. dt is in the format YYYY-MM-DD
    dt = pd.to_datetime(dt)
    
    #Check if the _date is within 7 days of the dt
    if np.abs((_date - dt).days) <= 7:
        return True
    else:
        return False

if __name__ == "__main__":
    # Base directory
    base_dir_pattern = "../gis-stac/IA_sentinel1/2020-*"
    base_dirs = glob.glob(base_dir_pattern)
    
    #Sort the directories
    base_dirs.sort()
    
    #retrieve the first directory and the first file in the directory
    first_dir = base_dirs[0]
    _files = glob.glob(f"{first_dir}/*.tif")
    
    for _f in _files:
        print(f"\nChecking file: {_f}\n")
        if check_date(_f, "2020-04-05"):
            print("Within 7 days")
        else:
            print("More than 7 days")
    
    
    
    