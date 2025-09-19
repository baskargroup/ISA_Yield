# Instructions

1. Run ```python yld_proc/save_parquet_chunks.py``` to convert yearly csv file into chunks of parquet files. 
2. Run ```python yld_proc/data_processing.py``` to combine chunks into one parquet file.
3. Run ```python yld_proc/generate_yield_geotiffs.py``` to generate yield geotiffs for each field. 
4. Run ```python generate_data_timeseries.py``` to generate modality data corresponding to each of the fields in the yield data. Use _xr to save xarray format data.
5. Run ```python process_ts.py``` to process all the modality data and save field-level data. Use `ts` on 98, 177 to select number of timpoints for the dataset.
6. Run ```python gen_splits.py``` to generate the train.txt, valid.txt, and test.txt files (to split the data during training, validation, and testing)