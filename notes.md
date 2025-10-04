# Instructions

1. Run ```python yld_proc/save_parquet_chunks.py``` to convert yearly csv file into chunks of parquet files. 
2. Run ```python yld_proc/data_processing.py``` to combine chunks into one parquet file.
3. Run ```python yld_proc/generate_yield_geotiffs.py``` to generate yield geotiffs for each field. 
4. Run ```python generate_data_timeseries.py``` to generate modality data corresponding to each of the fields in the yield data. Only generates data where all the modalities are available for a particular date. Check the used_dates_log.txt for dates used. Use _xr to save xarray format data.
5. Run ```python process_ts.py``` to process all the modality data and save field-level data. Use `ts` on 98, 177 to select number of timpoints for the dataset.
    a. Run ```python process_ts_monthly.py``` or  ```python process_ts_biweek.py``` as needed.
6. Run ```python gen_splits.py``` to generate the train.txt, valid.txt, and test.txt files (to split the data during training, validation, and testing)

# Downstream Training - Terramind

1. 