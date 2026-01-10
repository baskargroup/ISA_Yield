# Instructions

1. Run ```python yld_proc/save_parquet_chunks.py``` to convert yearly csv file into chunks of parquet files. 
2. Run ```python yld_proc/data_processing.py``` to combine chunks into one parquet file.
3. Run ```python yld_proc/generate_yield_geotiffs.py``` to generate yield geotiffs for each field. 
4. Run ```python generate_weather_chloe.py```to generate 10m resolution weather TIF files:
   - Input: 1km Daymet weather data
   - Process: For each field, identify overlapping 1km weather pixels, compute spatial mean, and resample to field-level 10m resolution
   - Output: Field-aligned weather TIF files
5. Run ```python generate_data_timeseries_new_chloe.py``` to generate modality data corresponding to each of the fields in the yield data. Only generates data where all the modalities are available for a particular date. Check the used_dates_log.txt for dates used. Use _xr to save xarray format data.
6. Run ```python process_ts_weekly_chloe.py``` to process all the modality data and save field-level data.
7. Run ```python gen_splits.py``` to generate the train.txt, valid.txt, and test.txt files (to split the data during training, validation, and testing)

# Downstream Training - Terramind

1. Use ```terratorch fit -c configs/config_s1.yaml``` or other configs as needed to train.
2. Use ```terratorch test -c configs/config_s12c.yaml --ckpt output/S12c/lightning_logs/version_0/checkpoints/epoch\=329-step\=4950.ckpt ``` or other configs and ckpts as needed.
3. The experiment statistics will be logged to wandb