import glob
# from datetime import datetime

# base_dir_pattern = "../gis-stac/IA_sentinel2/2019-*"
# base_dirs = glob.glob(base_dir_pattern)

# #Sort the directories
# base_dirs.sort()

# # Specify processing start date
# start_dt = "2019-08-18"
# start = datetime.strptime(start_dt, "%Y-%m-%d")

# # Remove directories that are less than the start date
# base_dirs = [dir for dir in base_dirs if datetime.strptime(dir.split("/")[-1],"%Y-%m-%d") >= start]

# print(f"Found {len(base_dirs)} directories after {start_dt}")

# # Print the directories
# for dir in base_dirs:
#     print(dir)


import pandas as pd

 #Read the start and end date from the "sundays_and_saturdays_2004_2024.csv" file
_dates = pd.read_csv("sundays_and_saturdays_2004_2024.csv")
#First column is the start date and the second column is the end date. Pick only those rows where the start date is in 2020, 2021, 2022, 2023, 2024
_dates = _dates[_dates["Sunday"].str.contains("2019")]
#Pick only those _dates where the start date is greater than or equal to 2019-08-18
_dates = _dates[_dates["Sunday"] >= "2019-08-18"]
#Get the start and end dates
_start_dates = _dates["Sunday"].values
_end_dates = _dates["Saturday"].values

print(f"Number of dates: {len(_start_dates)}")
print(f"Start Dates: {_start_dates}")
print(f"End Dates: {_end_dates}")