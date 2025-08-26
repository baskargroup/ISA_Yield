"""
Folder A has the following structure:

A
 - Week_1
    - <code>.A<Year+Julian Day>.<Other terms>tif
    - <code>.A<Year+Julian Day>.<Other terms>tif
 - Week_2
    - <code>.A<Year+Julian Day>.<Other terms>tif
    - <code>.A<Year+Julian Day>.<Other terms>tif
    
 ...
    - Week_n
        - <code>.A<Year+Julian Day>.<Other terms>tif
        - <code>.A<Year+Julian Day>.<Other terms>tif

Folder B has the following structure:

B
    - Week_1
        - S2A_MSIL2A_20190413T171901_R012_T14TPM_20201007T013637_B01.tif
        - ...
    - Week_2
        - ...
        - ...
    ...
    - Week_n
            - ...
            - ...

For a given week in Folder B, find all unique dates in Folder B by parsing the filenames. Find matching dates in Folder A for the corresponding week. For each matching date, copy the corresponding file from Folder A to a new directory. Create a report for the matching dates.

"""

import os
import glob
import shutil
from datetime import datetime

def get_unique_datesA(files):
     for dir_b in os.listdir(files):
        print(f"Processing directory: {dir_b}")
        # Find all files in the directory
        files_b = glob.glob(os.path.join(files, dir_b, "*.tif"))
        
        if len(files_b) == 0:
            print(f"No tiff files found for directory: {dir_b}")
            continue
        else:
            #Extract Date from the file name
            dates_b = [os.path.basename(f).split(".")[1].split('A')[1] for f in files_b if not f.startswith('Band')]
            # The above list has duplicates, so we will convert it to a set to get unique dates
            dates_b = set(dates_b)
            
            #dates_a = [convert_to_julian(datetime.strptime(f, "%Y%m%d").date()) for f in dates_b]
            
        return dates_b


def get_unique_datesB(files):
     for dir_a in os.listdir(files):
        print(f"Processing directory: {dir_a}")
        # Find all files in the directory
        files_b = glob.glob(os.path.join(files, dir_a, "*.tif"))
        #print(os.path.basename(files_a[0]).split(".")[0].split("_")[2])
        #Extract Date from the file name
        dates_b = [os.path.basename(f).split(".")[0].split("_")[2].split('T')[0] for f in files_b]
        # The above list has duplicates, so we will convert it to a set to get unique dates
        dates_b = set(dates_b)
        
        dates_a = [convert_to_julian(datetime.strptime(f, "%Y%m%d").date()) for f in dates_b]
        
        return dates_a

def convert_to_julian(dt):
    """
    Convert a date to julian day.
    """
    return dt.strftime("%Y%j")

if __name__ == "__main__":

    # Define the base directories
    folder_a = "/work/mech-ai-scratch/rtali/gis-stac/IA_modis_NBAR"
    folder_b = "/work/mech-ai-scratch/rtali/gis-stac/IA_sentinel2"
    
    #unique_b = get_unique_datesB(folder_b)
    #unique_a = get_unique_datesA(folder_a)
    
    # Keep only the dates that are common in both lists
    #common_dates = unique_b.intersection(unique_a)
    #print(f"Common dates: {common_dates}")
    
    #print(f"Unique dates in Folder A: {unique_a}")
    
    listB = []
    
    for dir_b in os.listdir(folder_b):
        
        """
        If the dirname begins with 2020, 2021, 2022, 2023
        """
        if dir_b.startswith("2020") or dir_b.startswith("2021") or dir_b.startswith("2022") or dir_b.startswith("2023"):
            
            print(f"Processing directory: {dir_b}")
            # Find all files in the directory
            files_b = glob.glob(os.path.join(folder_b, dir_b, "*.tif"))
            #print(os.path.basename(files_a[0]).split(".")[0].split("_")[2])
            #Extract Date from the file name
            dates_b = [os.path.basename(f).split(".")[0].split("_")[2].split('T')[0] for f in files_b]
            # The above list has duplicates, so we will convert it to a set to get unique dates
            dates_b = set(dates_b)
            
            dates_a = [convert_to_julian(datetime.strptime(f, "%Y%m%d").date()) for f in dates_b]
            
            listB.append(dates_a)
        
    print("\nDone\n")
    
    listA = []
    for dir_bx in os.listdir(folder_a):
        
        if dir_bx.startswith("2020") or dir_bx.startswith("2021") or dir_bx.startswith("2022") or dir_bx.startswith("2023"):
        
            print(f"Processing directory: {dir_bx}")
            # Find all files in the directory
            files_bx = glob.glob(os.path.join(folder_a, dir_bx, "*.tif"))
            if len(files_bx) == 0:
                print(f"No tiff files found for directory: {dir_bx}")
                listA.append([])
                continue
            else:
                #print(os.path.basename(files_bx[0]).split(".")[1].split('A')[1])
                #Extract Date from the file name
                try:
                    dates_bx = [os.path.basename(f).split(".")[1].split('A')[1] for f in files_bx if not os.path.basename(f).startswith('Band')]
                    dates_bx = set(dates_bx) # Dedup
                    
                    listA.append(dates_bx)
                except Exception as e:
                    print(f"Error: {e}")
            
        
    print("\nDone\n")
    
    # Keep only the dates that are common in both lists
    
    """
    common_dates = []
    """
    
    assert len(listA) == len(listB)
    
    for i in range(len(listA)):
        common_dates = set(listA[i]).intersection(set(listB[i]))
        print(f"Common dates: {common_dates}")
        print(f"Number of common dates: {len(common_dates)}")




