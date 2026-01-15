# Soil Moisture Processing Changes

## Summary of Changes

### 1. Monthly Processing with Parquet Caching
- **Changed from**: Decade-based batch processing
- **Changed to**: Month-by-month processing with individual parquet files
- **Benefit**: Incremental updates - only process new months when re-running with updated NetCDF

### 2. File Naming Convention
- Monthly cache files: `soil_moisture_YYYY_MM.parquet` (e.g., `soil_moisture_2024_03.parquet`)
- Final output CSVs: `soil_moisture_YYYY_YYYY.csv` (decadal, e.g., `soil_moisture_2020_2029.csv`)

### 3. Smart Caching Logic
- Checks for existing parquet files before processing
- Skips months that are already cached
- Only processes months >= START_DATE where cache doesn't exist
- Enables re-running with newer NetCDF data to append only new months

### 4. Dynamic End Date
- **config.END_DATE = None**: Uses last available month in NetCDF time series
- **config.END_DATE = "YYYY-MM-DD"**: Processes up to specified date
- Automatically adapts to NetCDF temporal extent

### 5. Area-Weighted Zonal Statistics
The `area_weighted_mean_optimized` function correctly implements:
- **Pixel-polygon intersection**: Calculates exact intersection area for edge pixels
- **Weighted averaging**: `sum(value * intersection_area) / sum(intersection_area)`
- **Spatial clipping**: Only processes pixels within polygon bounding box
- **NoData handling**: Excludes NoData pixels from calculations

### 6. Processing Flow
```
1. Open NetCDF and determine time range (START_DATE to END_DATE or last month)
2. Pre-project polygons to NetCDF CRS (one-time operation)
3. Check existing cached parquet files
4. For each month in time range:
   - Skip if parquet exists
   - Load month raster
   - Broadcast to workers
   - Compute area-weighted mean per polygon
   - Save to parquet: cache/soil_moisture_YYYY_MM.parquet
5. Read all parquet files
6. Group by decade and save CSVs: soil_moisture_YYYY_YYYY.csv
```

### 7. Configuration Changes
```python
# config.py
CACHE_DIR: Path = OUTPUT_DIR / "cache"  # New: separate cache directory
END_DATE = None  # Changed: None = use NetCDF extent
```

### 8. Key Functions
- `get_parquet_filename(output_dir, year, month)`: Generate cache filename
- `get_existing_months(output_dir)`: Scan for existing cache files
- `area_weighted_mean_optimized(geom, raster_np, transform, nodata)`: Exact zonal stats
- `process_polygon_block_lazy(...)`: Dask worker for polygon blocks

### 9. Validation Notes
The area-weighted zonal statistics implementation is correct:
- Uses affine transform for pixel-to-coordinate conversion
- Calculates exact intersection areas using Shapely geometry operations
- Properly handles partial pixel coverage at polygon boundaries
- Excludes NoData values before averaging

### 10. Re-run Capability
When a more recent NetCDF is available:
1. Update `ROOT_ZONE_SOIL_MOISTURE_RELATIVE` path in config
2. Re-run script
3. Only new months (not in cache) will be processed
4. All parquet files (old + new) aggregated to updated decadal CSVs
