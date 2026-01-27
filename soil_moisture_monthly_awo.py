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
from dask import delayed
from pathlib import Path
import logging
import requests
from urllib.parse import urlencode
from datetime import datetime
import xml.etree.ElementTree as ET

from exactextract import exact_extract
from tools.dask import start_dask
from tools.logging_setup import setup_logging
from config import soil_moisture_cfg as config
from dask.distributed import as_completed
import time

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
    """
    Compute zonal statistics for a polygon block using exact_extract.

    Args:
        raster_data (xarray.DataArray): The raster data for a specific time slice.
        gdf_block_future (GeoDataFrame): A subset of polygons to process.
        year (int): The year of the data.
        month (int): The month of the data.
        unique_id (str): The column name for the unique polygon identifier.

    Returns:
        tuple: (year, month, list of (uid, mean_value) tuples)
    """
   
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
    """
    Write monthly results to parquet with atomic file operations.

    Args:
        month_results (list): List of results from process_polygon_block_lazy.
        cache_dir (str or Path): Directory to save the parquet file.
        variable_name (str): Name of the variable being processed.
        unique_id (str): The column name for the unique polygon identifier.

    Returns:
        str: Path to the saved file or error message.
    """
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


def get_ncss_time_range(base_url):
    """
    Query NCSS metadata to get the valid time range of the dataset.

    Args:
        base_url (str): The base URL of the NCSS service.

    Returns:
        str: The end date string (ISO format) if found, else None.
    """
    try:
        response = requests.get(f"{base_url}/dataset.xml", timeout=10)
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        # Search for <end> tag anywhere in the tree (ignoring namespaces)
        for elem in tree.iter():
            if elem.tag.endswith("end") and elem.text:
                return elem.text
    except Exception as e:
        logger.warning(f"Could not fetch NCSS metadata: {e}")
    return None


def download_mdb_soilmoisture_subset(
    polygon_path,
    output_path,
    start_date,
    cache_dir,
    end_date=None,
    gdf=None,
    unique_id=None,
):
    """
    Downloads a subset of the soil moisture NetCDF from NCI THREDDS server
    based on the bounding box of the provided polygon file.

    Args:
        polygon_path (Path): Path to the polygon shapefile.
        output_path (Path): Path where the downloaded NetCDF will be saved.
        start_date (str): Start date for the data subset (YYYY-MM-DD).
        cache_dir (Path): Directory containing cached parquet files (used for date checking).
        end_date (str, optional): End date for the data subset. Defaults to None (latest).
        gdf (GeoDataFrame, optional): Pre-loaded GeoDataFrame to avoid reloading.
        unique_id (str, optional): Column name for unique ID, used if loading gdf.

    Returns:
        GeoDataFrame: The loaded polygon data (if loaded during process), else None.
    """
    output_path = Path(output_path)
    base_url = config.THREDDS_AWO_ROOT_ZONE_SOIL_MOISTURE_BASE_URL

    # Check remote availability against cache
    meta_end_str = get_ncss_time_range(base_url)
    should_download = True

    if not output_path.exists():
        logger.info(f"Local NetCDF file missing at {output_path}. Downloading...")
        should_download = True
    elif meta_end_str:
        remote_end_dt = pd.to_datetime(meta_end_str)

        # If remote date is timezone-aware (e.g., from 'Z' suffix), make it naive.
        # This prevents 'Cannot compare tz-naive and tz-aware timestamps' errors
        # when comparing against the local NetCDF time, which is naive.
        if remote_end_dt.tz:
            remote_end_dt = remote_end_dt.tz_localize(None)

        # Check if local file is already up to date
        local_is_current = False
        try:
            with xr.open_dataset(output_path, chunks={}) as ds:
                local_end_dt = pd.to_datetime(ds.time.values[-1])

            # Compare year/month only to avoid day mismatch (e.g. 1st vs 31st)
            if (local_end_dt.year, local_end_dt.month) >= (
                remote_end_dt.year,
                remote_end_dt.month,
            ):
                local_is_current = True
                logger.info(
                    f"Local NetCDF up to date ({local_end_dt:%Y-%m-%d}). Skipping download."
                )
        except Exception as e:
            logger.warning(f"Could not check local NetCDF date: {e}")

        if local_is_current:
            should_download = False
        else:
            remote_ym = (remote_end_dt.year, remote_end_dt.month)
            existing_months = get_existing_months(cache_dir)
            if existing_months:
                last_cached_ym = max(existing_months)

                if remote_ym < last_cached_ym:
                    logger.error(
                        f"ALERT: Remote NetCDF end date ({remote_end_dt:%Y-%m-%d}) is earlier than most recent cached result ({last_cached_ym}). Potential date alignment problem!"
                    )
                    should_download = False
                elif remote_ym == last_cached_ym:
                    logger.info(
                        f"Remote NetCDF end date ({remote_end_dt:%Y-%m-%d}) already in local cache. Skipping download."
                    )
                    should_download = False
                else:
                    logger.info(
                        f"New data available (Remote: {remote_ym} > Cache: {last_cached_ym}). Downloading soil moisture data..."
                    )
    else:
        logger.info(
            f"NetCDF file exists at {output_path} and metadata unavailable, skipping download."
        )
        should_download = False

    if not should_download:
        return gdf

    logger.info(f"Downloading soil moisture data from NCI THREDDS to {output_path}...")

    try:
        # 1. Get bounds from shapefile
        if gdf is None:
            cols = [unique_id, "geometry"] if unique_id else None
            gdf = gpd.read_file(polygon_path, columns=cols)

        # Transform to WGS84 (EPSG:4326) for NCSS lat/lon coordinates
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf_bounds = gdf.to_crs("EPSG:4326")
        else:
            gdf_bounds = gdf

        minx, miny, maxx, maxy = gdf_bounds.total_bounds

        # Buffer slightly (0.1 degree) to ensure full coverage of edge polygons
        buffer = 0.1
        minx = max(-180, minx - buffer)
        miny = max(-90, miny - buffer)
        maxx = min(180, maxx + buffer)
        maxy = min(90, maxy + buffer)

        # 2. Construct NCSS URL

        # Format dates to YYYY-MM-DDTHH:MM:SSZ
        def format_date(d_str):
            if "T" not in d_str:
                return f"{d_str}T00:00:00Z"
            return d_str

        s_date = format_date(start_date)

        if end_date:
            e_date = format_date(end_date)
        else:
            # Try to get actual dataset end date, fallback to now
            e_date = (
                meta_end_str
                if meta_end_str
                else datetime.now().strftime("%Y-%m-%dT00:00:00Z")
            )

        params = {
            "var": "sm_pct",
            "north": maxy,
            "west": minx,
            "east": maxx,
            "south": miny,
            "horizStride": 1,
            "time_start": s_date,
            "time_end": e_date,
            "accept": "netcdf4-classic",
        }

        url = f"{base_url}?{urlencode(params)}"
        logger.info(f"Requesting subset: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Download complete.")
        return gdf

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def load_and_scatter_polygons(dask_client, polygon_path, target_crs, unique_id, block_size, gdf=None):
    """
    Loads (or uses existing) polygons, projects them, splits into blocks, and scatters to Dask workers.
    Returns a list of Dask futures for the blocks.
    """
    logger.info("Preparing polygon blocks for scatter...")
    
    # Load if not provided
    if gdf is None:
        logger.info(f"Reading shapefile {polygon_path}...")
        gdf = gpd.read_file(polygon_path, columns=[unique_id, "geometry"])
    
    # Project if needed
    if gdf.crs and gdf.crs.to_epsg() != target_crs.to_epsg():
        logger.info(f"Projecting polygons to {target_crs}...")
        gdf = gdf.to_crs(target_crs)
        
    # Split into blocks
    logger.info(f"Splitting {len(gdf)} polygons into blocks of {block_size}...")
    total_polygons = len(gdf)
    blocks = [gdf.iloc[i : i + block_size].copy() for i in range(0, total_polygons, block_size)]
    
    # Scatter
    logger.info("Scattering blocks to workers...")
    futures = [dask_client.scatter(block) for block in blocks]
    
    # Explicit cleanup
    del gdf
    del blocks
    gc.collect()
    
    return futures

def compute_zonal_statistics(
    polygon_shapefile,
    unique_id,
    cache_dir,
    output_dir,
    netcdf_file,
    variable_name="soil_moisture",
    n_workers=4,
    block_size=1000,
    batch_size=12,
    gdf=None,
):
    """
    Compute monthly zonal statistics from NetCDF to parquet cache.
    Processes only months not already cached. Uses Dask for parallel computation.
    Polygons are often smaller than the 5km raster pixels so we keep polygons as vector geometry
    by using "Exact extract" library for zonal stats.
    This is CPU intensive so more workers is better if you have the resources
    
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
        gdf (GeoDataFrame, optional): Pre-loaded GeoDataFrame.

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

    # Initialize Dask distributed client
    dask_client = start_dask(workers=n_workers)

    # Initial scatter
    try:
        gdf_block_futures = load_and_scatter_polygons(
            dask_client, polygon_shapefile, target_crs, unique_id, block_size, gdf
        )
    except Exception as e:
        logger.error(f"Failed to prepare polygons: {e}")
        ds.close()
        return return_value

    # Clean up local gdf if it was passed in
    if gdf is not None:
        del gdf
    gc.collect()
    # Process months in batches to control memory usage
    for batch_start in range(0, len(job_list), batch_size):
        batch = job_list[batch_start:batch_start + batch_size]
        write_tasks = []

        # Retry logic to handle worker crashes (e.g. OOM) or lost scattered data
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info("Retrying batch: Re-reading and re-scattering polygon blocks...")
                    gdf_block_futures = load_and_scatter_polygons(
                        dask_client, polygon_shapefile, target_crs, unique_id, block_size, gdf=None
                    )

                # Process each month in the batch
                for year, month, i in batch:
                    # Load single month raster and scatter to workers. Reduces dask schedular overhead.
                    # faster because it preps shared in-memory arrays for Exact_extract workers that are not dask aware.
                    month_da = da.isel(time=i).squeeze().compute()
                    month_future = dask_client.scatter(month_da)

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
                futures = dask_client.compute(write_tasks, sync=False)

                # Log progress as tasks complete
                completed_count = 0
                for future in as_completed(futures):
                    completed_count += 1
                    logger.info(f"Progress: {completed_count}/{len(futures)} months completed")

                # Gather results
                written_files = dask_client.gather(futures)

                # Force garbage collection on all workers and locally
                dask_client.run(gc.collect)
                gc.collect()

                # Log results
                for filepath in written_files:
                    if isinstance(filepath, str) and filepath.startswith("Failed"):
                        logger.error(filepath)
                    else:
                        logger.info(f"Saved {filepath}")
                        return_value = True
                break
            except Exception as e:
                logger.warning(f"Batch {batch_start//batch_size + 1} failed  {attempt + 1}/{max_retries}: {e}")
                dask_client.run(gc.collect)
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)
    

    # Cleanup scattered futures and close dataset
    del gdf_block_futures
    ds.close()
    gc.collect()

    return return_value


def aggregate_parquets(cache_dir, output_dir, unique_id, variable_name="soil_moisture"):
    """
    Aggregate monthly parquet files into decadal CSV zip files.
    
    Groups all cached monthly data by decade and writes compressed CSVs.

    Args:
        cache_dir (Path): Directory containing monthly parquet files.
        output_dir (Path): Directory to save aggregated zip files.
        unique_id (str): Column name for unique polygon ID.
        variable_name (str): Name of the variable to include in output.
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
    logger.info(f"Input: {config.LOCAL_ROOT_ZONE_SOIL_MOISTURE_RELATIVE_NETCDF_PATH}")
    logger.info(f"Polygons: {config.POLYGON_PATH}")
    logger.info(f"Date range: {config.START_DATE} to {config.END_DATE or 'end of file'}")

    gdf = None
    try:
        gdf = download_mdb_soilmoisture_subset(
            config.POLYGON_PATH,
            config.LOCAL_ROOT_ZONE_SOIL_MOISTURE_RELATIVE_NETCDF_PATH,
            config.START_DATE,
            config.CACHE_DIR,
            config.END_DATE,
            gdf=gdf,
            unique_id=config.POLY_UID,
        )
    except Exception as e:
        logger.warning(
            f"Could not download NetCDF: {e}. Proceeding with existing file if available."
        )

    try:
        if compute_zonal_statistics(
            polygon_shapefile=config.POLYGON_PATH,
            unique_id=config.POLY_UID,
            cache_dir=config.CACHE_DIR,
            output_dir=config.OUTPUT_DIR,
            netcdf_file=config.LOCAL_ROOT_ZONE_SOIL_MOISTURE_RELATIVE_NETCDF_PATH,
            variable_name=config.SM_VAR,
            n_workers=config.DASK_N_WORKERS_OVERRIDE,
            block_size=config.BLOCK_SIZE,
            batch_size=config.BATCH_SIZE,
            gdf=gdf,
        ):
            aggregate_parquets(config.CACHE_DIR, config.OUTPUT_DIR, config.POLY_UID, config.SM_VAR)
        logger.info("Processing complete")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
