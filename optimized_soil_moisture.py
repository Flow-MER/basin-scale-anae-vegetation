import os
import logging
import pandas as pd
import geopandas as gpd
import xarray as xr
#import rioxarray
from exactextract import exact_extract
from tqdm import tqdm
import zipfile
from tools.logging_setup import setup_logging
from pathlib import Path
from config import soil_moisture_cfg as config

# Initialize Environment
logger = logging.getLogger(__name__)


def process_soil_moisture_exact():
    """
    Compute monthly zonal statistics from a NetCDF raster time series.
    
    Optimizations:
    - Batch processing in chunks to reduce I/O
    - Memory-efficient data handling
    - Vectorized date operations
    """
    crs_fallback="EPSG:4326"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load and validate zones
    zones = gpd.read_file(config.POLYGON_PATH)
    if zones.crs is None:
        raise ValueError("Zones file must have a defined CRS")

    # Open dataset with chunking for memory efficiency
    with xr.open_dataset(config.ROOT_ZONE_SOIL_MOISTURE_RELATIVE, chunks={'time': config.CHUNK_SIZE}) as ds:
        ds = ds.sel(time=slice(config.START_DATE, config.END_DATE))
        
        if config.SM_VAR not in ds:
            raise ValueError(f"Variable '{config.SM_VAR}' not found in dataset.")
        da = ds[config.SM_VAR]

        # Optimize dimension handling
        dim_mapping = {}
        if "longitude" in da.dims:
            dim_mapping["longitude"] = "x"
        if "latitude" in da.dims:
            dim_mapping["latitude"] = "y"
        if dim_mapping:
            da = da.rename(dim_mapping)

        # Set CRS and reproject zones
        if not da.rio.crs:
            da = da.rio.write_crs(crs_fallback)
        zones = zones.to_crs(da.rio.crs)

        # Clip to bounds
        da = da.rio.clip_box(*zones.total_bounds)

        # Process in chunks and collect results
        output_file = config.OUTPUT_DIR / f"soil_moisture_zonal_{config.START_DATE}_{config.END_DATE}.csv"
        results = []
        
        for i in tqdm(range(len(da.time)), desc="Processing time slices"):
            slice_da = da.isel(time=i)
            stats = exact_extract(slice_da, zones, "mean", include_cols=[config.POLY_UID], output="pandas")
            stats["date"] = pd.to_datetime(da.time.values[i]).replace(day=1)
            results.append(stats)
            
            # Write in chunks to manage memory
            if len(results) >= config.CHUNK_SIZE or i == len(da.time) - 1:
                chunk_df = pd.concat(results, ignore_index=True)
                chunk_df.to_csv(output_file, mode="a", header=(i < config.CHUNK_SIZE), index=False)
                results = []

    return output_file

def main():
    setup_logging(config.LOG_DIR, Path(__file__).name)

    csv_path = process_soil_moisture_exact()

    print(f"Output saved to: {csv_path}")
    
    # Compress output
    zip_file = Path(str(csv_path) + ".zip")
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(csv_path, str(csv_path))
    print(f"Compressed to: {zip_file}")

if __name__ == "__main__":
    main()