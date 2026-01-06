# Change Log

## Modified Files (3 files)

### 1. `generate_data_timeseries_new_chloe.py`
**Purpose:** Enhanced spatial alignment and resampling for geospatial data processing

**Changes:**
- Added import: `from rasterio.warp import reproject, Resampling`
- **New Function:** `resample_to_reference()` - Resamples source geospatial data to match a reference grid (CRS, affine transform, shape) using bilinear interpolation
  - Handles multi-band raster data
  - Provides error handling for resampling failures
- **Modified Function:** `load_static_data()`
  - Updated signature to include reference grid parameters: `ref_transform`, `ref_crs`, `ref_shape`
  - Increased default `repeat_times` from 12 to 24
  - Added spatial alignment step using new `resample_to_reference()` function before temporal repetition
  - Enhanced documentation to reflect alignment capability

**Rationale:** Ensures all static geospatial data is properly aligned to a consistent reference grid before processing, improving data consistency and model input quality.

---

### 2. `generate_data_timeseries_withoutcombine_chloe.py`
**Purpose:** Data validation and spatial alignment for time-series data generation

**Changes:**
- Added import: `from rasterio.io import MemoryFile`
- Added import: `from rasterio.warp import reproject, Resampling` (implied by resampling function)
- **New Function:** `resample_to_reference()` - Same spatial alignment function as in file #1
  - Resamples multi-band geospatial data to reference grid
  - Uses bilinear interpolation for accurate spatial transformation
  - Returns None on error for graceful failure handling
- **Removed:** `get_bbox_from_geotiff()` - Replaced by reference grid-based approach
- **Modified Function:** `is_valid_date_combined()`
  - Changed from mask-based validation to resample-based validation
  - Updated signature: replaced `bbox_gdf` parameter with `ref_transform`, `ref_crs`, `ref_shape`
  - Enhanced validation logic to test data resampling capability instead of bounding box masking
  - Improved data consistency checking

**Rationale:** Shifts from bounding-box-based cropping to proper spatial alignment using reference grids, enabling multi-modal data harmonization without data loss.

---

### 3. `process_ts_weekly_chloe.py`
**Purpose:** Multi-modal temporal aggregation and improved date tracking

**Changes:**
- **Modified Function:** `get_monthly_indices()`
  - Added parameter: `modality_name` (string)
  - Updated logic to retrieve modality-specific dates from nested dictionary structure
  - Changed from flat list lookup to nested dictionary: `used_dates_dict[filename][modality]`
  - Enables per-modality temporal aggregation

- **Modified Function:** `get_weekly_indices()`
  - Added parameter: `modality_name` (string)
  - Updated logic for modality-specific date retrieval
  - Changed from flat list to nested dictionary: `used_dates_dict[filename][modality_name.upper()]`
  - Enables per-modality weekly aggregation

- **Modified Function Call:** `process_single_file()`
  - Updated `get_weekly_indices()` call to pass `modality_name` parameter
  - Line ~406: Added modality_name argument to function call

- **Modified Function:** `parse_used_dates_log()`
  - Changed return structure from flat dict `{filename: [dates]}` to nested dict `{filename: {modality: [dates]}}`
  - Updated parsing logic to handle new hierarchical log format:
    - Identifies file header lines: `"ST2021IA0013_Corn.tif (Field: ...)"`
    - Identifies modality lines: `"  S2L2A: 15 dates"`
    - Parses date lines: `"    2021-04-15, 2021-05-01, ..."`
  - Enhanced with state tracking: `current_file` and `current_modality` variables
  - Log file reference updated: `processing_log.txt` → `dates_log.txt`

- **Updated Log Path:** Line ~621
  - Changed from: `processing_log.txt`
  - Changed to: `dates_log.txt`

**Rationale:** Enables per-modality temporal tracking and aggregation, allowing different sensors (S2L2A, S1RTC, WEATHER, etc.) to have independent temporal sampling and aggregation strategies.

---

## Summary

These changes implement a **unified spatial harmonization and multi-modal temporal tracking system**:

1. **Spatial Harmonization:** All geospatial data is resampled to a consistent reference grid using proper coordinate transformation
2. **Multi-Modal Support:** Temporal aggregation now works independently per modality, allowing different sensors with different acquisition schedules
3. **Improved Robustness:** Reference-grid-based approach replaces bounding-box masking for better data integrity
4. **Enhanced Logging:** Structured date tracking per modality enables better reproducibility and debugging

---

**Date:** January 6, 2026
**Branch:** (current)
**Status:** Ready for commit
