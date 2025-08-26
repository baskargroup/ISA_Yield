from datetime import datetime
import os
import pandas as pd
import numpy as np
import glob

def convert_to_julian(dt):
    """
    Convert a date to julian day.
    """
    return dt.strftime("%Y%j")

def julian_to_date(julian):
    """
    Convert a julian day to date.
    """
    return datetime.strptime(julian, "%Y%j").date()


def check_date(file_name, dt):
    _date = file_name.split(".")[1]
    _date = _date.split("A")[1]
    _date = julian_to_date(_date)
    
    print(f"Date: {_date}")
    
    #Convert to datetime. _date is a string in the format YYYY-MM-DD
    _date = pd.to_datetime(_date, format="%Y-%m-%d")
    
    #Convert dt to datetime. dt is in the format YYYY-MM-DD
    dt = pd.to_datetime(str(dt))
    
    #Check if the _date is within 7 days of the dt
    if np.abs((_date - dt).days) <= 7:
        return True
    else:
        return False

if __name__ == "__main__":
    #print(julian_to_date("2020135"))
    file_name = "hhh/MOD09A1.A2020135.h08v05.006.2020149225048.tif"
    
    print(os.path.basename(file_name).replace(".tif", ""))
    
    dt = "2020-05-30"
    #print(check_date(file_name, dt))
    