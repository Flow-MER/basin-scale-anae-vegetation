"""
Dask-parallel zonal statistics for NetCDF soil moisture.
Monthly processing with parquet caching for incremental updates.
Features: Lazy loading, pre-projection, and broadcasted rasters.
"""

import geopandas as gpd
import gc
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.prepared import prep
from dask import delayed
from pathlib import Path
import logging

from exactextract import exact_extract
from tools.dask import start_dask
from tools.logging_setup import setup_logging
from config import soil_moisture_cfg as config
from dask.distributed import as_completed

logger = logging.getLogger(__name__)

def get_parquet_filename(output_dir, year, month):
    """Generate parquet filename for a given year and month."""
    return output_dir / f"soil_moisture_{year}_{month:02d}.parquet"

def get_existing_months(output_dir):
    """Return set of (year, month) tuples for existing parquet files."""
    existing = set()
    for f in output_dir.glob("soil_moisture_*.parquet"):
        try:
            parts = f.stem.split('_')
            year, month = int(parts[2]), int(parts[3])
            existing.add((year, month))
        except (IndexError, ValueError):
            continue
    return existing

@delayed
def process_polygon_block_lazy(raster_data, gdf_block_future, year, month, unique_id):
    """Compute zonal statistics for a polygon block using exact_extract."""
   
    # Extract mean values for all polygons in block
    stats_df = exact_extract(
        raster_data, 
        gdf_block_future, 
        ['mean'], 
        include_cols=[unique_id], 
        output='pandas',
    )
    
    # Convert to list and clear DataFrame from memory
    val_col = [c for c in stats_df.columns if c != unique_id][0]
    results = list(zip(stats_df[unique_id], stats_df[val_col]))
    del stats_df
    return year, month, results

@delayed
def write_month_parquet(month_results, cache_dir, variable_name, unique_id):
    """Write monthly results to parquet with atomic file operations."""
    year, month = month_results[0][:2]
    # Flatten all block results into single list
    data = [(uid, val) for _, _, block in month_results for uid, val in block]
    
    df = pd.DataFrame(data, columns=[unique_id, variable_name])
    df['year'] = year
    df['month'] = month
    
    parquet_path = get_parquet_filename(Path(cache_dir), year, month)
          
    # Atomic write: tmp file then rename
    try:
        tmp_out = parquet_path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp_out, index=False)  
        tmp_out.rename(parquet_path)
        result = str(parquet_path)
    except Exception as e:
        tmp_out.unlink(missing_ok=True)
        result = f"Failed {year}-{month:02d}: {e}"
    finally:
        del df, data  # Explicit cleanup
    return result


# ------------------------------
# Main Computation logic
# ------------------------------

def compute_zonal_statistics(polygon_shapefile, unique_id, cache_dir, output_dir, netcdf_file, variable_name="soil_moisture", n_workers=4, block_size=1000, batch_size=12):
    """Compute monthly zonal statistics from NetCDF to parquet cache.
    Processes only months not already cached. Uses Dask for parallel computation.
    Exact extract zonal stats is CPU intensive so more workers is better if you have the resources
    
    Memory optimization strategy:
    - Process months in batches to limit concurrent memory usage
    - Scatter polygon blocks to workers once (avoid repeated transfers)
    - Explicit cleanup with del and gc.collect() after each batch
    - Use exact_extract for efficient zonal statistics
    
    Args:
        polygon_shapefile (str): Path to polygon shapefile
        unique_id (str): Column name for unique polygon ID
        cache_dir (str): Directory to store temporary parquet files
        output_dir (str): Directory for final CSV zip outputs
        netcdf_file (str): Path to input NetCDF file
        variable_name (str): Variable name in NetCDF to process
        n_workers (int): Number of Dask workers. 
        block_size (int): Number of polygons per worker block
        batch_size (int): Number of months to process in parallel

    Returns:
        bool: True if new data was processed, False if all cached
    
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return_value = False
    
    # Open NetCDF with lazy loading (chunks prevent loading entire file)
    logger.info("Opening Soil Moisture NetCDF...")
    try:
        ds = xr.open_dataset(netcdf_file, chunks={"time": 1})
    except FileNotFoundError:
        logger.error(f"NetCDF file not found: {netcdf_file}")
        return return_value 
    except Exception as e:
        logger.error(f"Failed to open NetCDF {netcdf_file}: {e}")
        return return_value
    
    # Determine time range to process
    all_times = pd.to_datetime(ds.time.values)
    logger.info(f"Soil Moisture NetCDF time range: {all_times[0].strftime('%Y-%m-%d')} to {all_times[-1].strftime('%Y-%m-%d')}")
    start_date = pd.to_datetime(config.START_DATE)
    end_date = pd.to_datetime(config.END_DATE) if config.END_DATE else all_times[-1]
    
    da = ds[variable_name].sel(time=slice(start_date, end_date))
    times = pd.to_datetime(da.time.values)
    
    # Skip already processed months (incremental processing)
    existing_months = get_existing_months(cache_dir)
    job_list = [(pd.Timestamp(t).year, pd.Timestamp(t).month, i) 
                for i, t in enumerate(times) 
                if (pd.Timestamp(t).year, pd.Timestamp(t).month) not in existing_months]
    
    if not job_list:
        logger.info(f"No new months to process. All data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} are cached.")
        return return_value
    
    logger.info(f"Processing {len(job_list)} months: {job_list[0][:2]} to {job_list[-1][:2]}")
    
    
    # Ensure CRS is set for raster data
    if da.rio.crs is None:
        da = da.rio.write_crs(config.CRS_FALLBACK)
    target_crs = da.rio.crs

    # Rechunk for optimal processing (512x512 spatial tiles, 1 time slice)
    x_dim = da.rio.x_dim
    y_dim = da.rio.y_dim    
    da = da.chunk({x_dim: 512, y_dim: 512, "time": 1})

    # Load and reproject polygons to match raster CRS
    logger.info(f"Projecting shapefile to {target_crs}...")
    try:
        gdf = gpd.read_file(polygon_shapefile, columns=[unique_id, "geometry"])
        gdf = gdf.to_crs(target_crs)
    except Exception as e:
        logger.error(f"Failed to read/project shapefile {polygon_shapefile}: {e}")
        ds.close()
        return return_value
    
    
    # Initialize Dask distributed client
    client = start_dask(workers=16)

    # Split GeoDataFrame into blocks and scatter to workers ONCE (avoid repeated transfers)
    logger.info("Splitting GeoDataFrame into memory-resident blocks...")
    total_polygons = len(gdf)
    gdf_blocks = [gdf.iloc[i : i + block_size].copy() for i in range(0, total_polygons, block_size)]
    gdf_block_futures = [client.scatter(block) for block in gdf_blocks]
    
    # Clean up local copies
    del gdf, gdf_blocks
    gc.collect()
    # Process months in batches to control memory usage
    for batch_start in range(0, len(job_list), batch_size):
        batch = job_list[batch_start:batch_start + batch_size]
        write_tasks = []
        
        # Process each month in the batch
        for year, month, i in batch:
            # Load single month raster and scatter to workers
            month_da = da.isel(time=i).squeeze().compute()
            month_future = client.scatter(month_da)
            
            # Create tasks for all polygon blocks for this month
            month_tasks = [
                process_polygon_block_lazy(month_future, block_fut, year, month, unique_id)
                for block_fut in gdf_block_futures
            ]
            
            # Chain write task after computation
            write_tasks.append(
                write_month_parquet(month_tasks, str(cache_dir), variable_name, unique_id)
            )
            
            # Clean up month data immediately
            del month_da
        
        # Execute batch: compute zonal stats + write parquet files in parallel
        logger.info(f"Computing batch {batch_start//batch_size + 1}: {len(write_tasks)} months")
        
        # Submit tasks and get futures for progress tracking
        futures = client.compute(write_tasks, sync=False)
        
        # Log progress as tasks complete
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            logger.info(f"Progress: {completed_count}/{len(futures)} months completed")
        
        # Gather results
        written_files = client.gather(futures)
        
        # Force garbage collection on all workers and locally
        client.run(gc.collect)
        gc.collect()
        
        # Log results
        for filepath in written_files:
            if isinstance(filepath, str) and filepath.startswith("Failed"):
                logger.error(filepath)
            else:
                logger.info(f"Saved {filepath}")
                return_value = True

    # Cleanup scattered futures and close dataset
    del gdf_block_futures
    ds.close()
    gc.collect()
    
    return return_value
 
def aggregate_parquets(cache_dir, output_dir, unique_id, variable_name="soil_moisture"):
    """Aggregate monthly parquet files into decadal CSV zip files.
    
    Groups all cached monthly data by decade and writes compressed CSVs.
    """
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Aggregating monthly parquets to decadal CSVs...")
    all_parquets = sorted(cache_dir.glob("soil_moisture_*.parquet"))
    if not all_parquets:
        logger.warning(f"No parquet files found in {cache_dir}")
        return
    
    # Read all parquet files and combine
    try:
        dfs = [pd.read_parquet(f) for f in all_parquets]
        combined = pd.concat(dfs, ignore_index=True)
        del dfs  # Free memory from list of DataFrames
        gc.collect()
    except Exception as e:
        logger.error(f"Failed to read parquet files: {e}")
        return
        
    # Group by decade and write separate files
    combined['decade'] = (combined['year'] // 10) * 10
    for decade, group in combined.groupby('decade'):
        start_year = decade
        end_year = decade + 9
        zip_path = output_dir / f"soil_moisture_{start_year}_{end_year}.zip"
        csv_name = f"soil_moisture_{start_year}_{end_year}.csv"
        try:
            # Write only required columns
            group[[unique_id, 'year', 'month', variable_name]].to_csv(
                zip_path,
                index=False,
                compression={'method': 'zip', 'archive_name': csv_name}
            )
            logger.info(f"Saved {zip_path}")
        except Exception as e:
            logger.error(f"Failed to write {zip_path}: {e}")
    
    # Final cleanup
    del combined
    gc.collect()

def main():
    """Process soil moisture NetCDF to decadal CSV outputs.
    
    Workflow:
    1. Compute monthly zonal statistics (cached as parquet)
    2. Aggregate cached months into decadal CSV zip files
    
    Configuration: see soil_moisture_cfg in config.py
    Logs: written to config.LOG_DIR
    """
    setup_logging(config.LOG_DIR, "soil_moisture_processing")
    logger.info("Starting soil moisture processing")
    logger.info(f"Input: {config.ROOT_ZONE_SOIL_MOISTURE_RELATIVE}")
    logger.info(f"Polygons: {config.POLYGON_PATH}")
    logger.info(f"Date range: {config.START_DATE} to {config.END_DATE or 'end of file'}")
    
    try:
        if compute_zonal_statistics(
            polygon_shapefile=config.POLYGON_PATH,
            unique_id=config.POLY_UID,
            cache_dir=config.CACHE_DIR,
            output_dir=config.OUTPUT_DIR,
            netcdf_file=config.ROOT_ZONE_SOIL_MOISTURE_RELATIVE,
            variable_name=config.SM_VAR,
            n_workers=config.DASK_N_WORKERS_OVERRIDE,
            block_size=config.BLOCK_SIZE,
            batch_size=config.BATCH_SIZE,
        ):
            aggregate_parquets(config.CACHE_DIR, config.OUTPUT_DIR, config.POLY_UID, config.SM_VAR)
        logger.info("Processing complete")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
    

if __name__ == "__main__":
    main()