# Running Instructions
0. Run ```source /work/mech-ai-scratch/rtali/gis-soil/soilenv/bin/activate``` to activate the required environment. 
1. Run ```python3 src/download_v2.py``` to download the dataset from Microsoft Planetary Computer.
2. Run ```python3 src/merge_v3.py``` to output datasets in EPSG:4326. Folder ```./final_s2_v3``` will be created.
3. Run ```python3 src/ai_ready.py``` to output xArray. One file per year will be created in ```./AI_ready``` folder.
4. Execute ```viz.ipynb``` to visualize the data and check the results