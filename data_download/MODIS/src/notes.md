# Instructions
0. Run ```python3 src/download_v2.py``` to download the data from Microsoft Planetary Computer
1. Run ```python3 src/reprojection_v2.py``` to reproject the data to EPSG:4326
2. Run ```python3 src/merge_v2.py``` to merge the data 
3. Final Data is under ```./modis```
4. Run ```python3 src/ai_ready.py``` to convert to xArray format. 1 file per year
5. Execute ```viz.ipynb``` to visualize the data and check the results