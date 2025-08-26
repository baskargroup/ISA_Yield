import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
from utilities import view_tif_files
import glob

if __name__ == "__main__":

    # Specify the glob pattern for the preview files
    _pattern = "./final_s2_v3_IA/2021-09-12/4326_*.tif"

    # Output folder
    plots_folder = "./plots/"

    # Call the function
    view_tif_files(glob.glob(_pattern), plots_folder)
