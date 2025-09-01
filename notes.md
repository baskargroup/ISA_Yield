# Instructions

1. Run ```python yld_proc/save_parquet_chunks.py``` to convert yearly csv file into chunks of parquet files. 
2. Run ```python yld_proc/data_processing.py``` to combine chunks into one parquet file.
3. Run ```python yld_proc/generate_yield_geotiffs.py``` to generate yield geotiffs for each field. 
4. Run ```python process_all.py``` to process all the modality data and save field-level data.