'''

import matplotlib.pyplot as plt
from rasterio.plot import show
import rasterio
import numpy as np
# from utilities import view_tif_files
import glob
import os


def view_tif_files(preview_files, output_folder="./plots"):
    for file in preview_files:
        try:
            with rasterio.open(file) as src:
                show(src)
                # Extract the filename without the extension and the path
                file_name = os.path.splitext(os.path.basename(file))[0]
                print(f"Plotting {file_name}")
                # Save the plot as a png file
                plt.savefig(f"{output_folder}/{file_name}.png")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue


if __name__ == "__main__":

    # Specify the glob pattern for the preview files
    _pattern = "./final_s1_IA/2021-04-18/4326_*.tif"

    # Output folder
    plots_folder = "./plots"

    # Call the function
    view_tif_files(glob.glob(_pattern), plots_folder)
'''


import matplotlib.pyplot as plt
import rasterio
import numpy as np
import glob
import os


def view_tif_files(preview_files, output_folder="./plots"):
    for file in preview_files:
        try:
            with rasterio.open(file) as src:
                # Read the raster data
                data = src.read(1)  # Read the first band
                # Create a new figure
                plt.figure()
                # Display the data using imshow with 'jet' colormap
                plt.imshow(data, cmap='jet')
                plt.colorbar()  # Add a colorbar for reference
                plt.title(os.path.basename(file))  # Set title to filename
                # Extract the filename without the extension and the path
                file_name = os.path.splitext(os.path.basename(file))[0]
                print(f"Plotting {file_name}")
                # Ensure output folder exists
                os.makedirs(output_folder, exist_ok=True)
                # Save the plot as a png file
                plt.savefig(f"{output_folder}/{file_name}.png")
                plt.close()  # Close the figure to free memory
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue


if __name__ == "__main__":
    # Specify the glob pattern for the preview files
    _pattern = "./final_s1_IA/2021-04-18/4326_*.tif"

    # Output folder
    plots_folder = "./plots"

    # Call the function
    view_tif_files(glob.glob(_pattern), plots_folder)
