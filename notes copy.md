# Data Preprocessing

## 1. Yield Data Processing
```bash
python yld_proc/save_parquet_chunks.py
```
Convert yearly CSV files into chunked parquet files.
```bash
python yld_proc/data_processing.py
```
Combine chunks into a single parquet file.
```bash
python extract_field_coordinates.py
```
Extract field boundaries into a CSV file to download geospatial data.
```bash
python yld_proc/generate_yield_geotiffs.py
```
Generate yield GeoTIFFs for each field.

## 2. Weather Data Processing
```bash
python generate_weather_chloe.py
```
Generate 10m resolution weather TIF files:
- **Input**: 1km Daymet weather data
- **Process**: Identify overlapping 1km pixels per field → compute spatial mean → resample to 10m
- **Output**: Field-aligned weather TIF files

## 3. Multimodal Data Generation
```bash
python generate_data_timeseries_new_chloe.py
```
Generate modality data for each field. Only includes dates where all modalities are available.
> Check `used_dates_log.txt` for dates used. Use `_xr` suffix to save xarray format.
```bash
python process_ts_weekly_chloe.py
```
Process and save field-level data for all modalities.

## 4. Train/Val/Test Split
```bash
python gen_splits.py
```
Generate `train.txt`, `valid.txt`, and `test.txt` for data splitting.



# Downstream Training - Terramind

## 1. Training
```bash
terratorch fit -c configs/config_s1.yaml
```
Train the model using the specified config file. Replace with other configs as needed.

## 2. Testing
```bash
terratorch test -c configs/config_s12c.yaml --ckpt output/S12c/lightning_logs/version_0/checkpoints/epoch\=329-step\=4950.ckpt
```
Evaluate the model with the specified config and checkpoint. Replace with other configs/ckpts as needed.

## 3. Logging
Experiment statistics are automatically logged to **wandb**.