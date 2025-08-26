import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
from utilitities import view_tif_files
import glob

if __name__ == "__main__":
   
    #Specify the glob pattern for the preview files
    _pattern = "./modis/2020-04-05/4326_*.tif"

    # Output folder
    plots_folder = "./plots/"

    # Call the function
    view_tif_files(glob.glob(_pattern), plots_folder)