"""
Dask-parallel zonal statistics for NetCDF soil moisture.
Monthly processing with parquet caching for incremental updates.
Features: Lazy loading, pre-projection, and broadcasted rasters.
"""

import geopandas as gpd
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.prepared import prep
from dask import delayed
from dask.distributed import Client
from pathlib import Path
import logging
import tempfile
import shutil

from tools.dask import start_dask
from tools.logging_setup import setup_logging
from config import soil_moisture_cfg as config

logger = logging.getLogger(__name__)

# ------------------------------
# Optimized Core Math
# ------------------------------

def area_weighted_mean_optimized(geom, raster_np, transform, nodata):
    """Compute area-weighted mean for polygon from raster.
    
    Returns np.nan if polygon doesn't intersect valid raster data.
    """
    minx, miny, maxx, maxy = geom.bounds
    inv_transform = ~transform
    col_min, row_max = inv_transform * (minx, miny)
    col_max, row_min = inv_transform * (maxx, maxy)

    r0, r1 = max(0, int(np.floor(row_min))), min(raster_np.shape[0], int(np.ceil(row_max)))
    c0, c1 = max(0, int(np.floor(col_min))), min(raster_np.shape[1], int(np.ceil(col_max)))
    
    if r0 >= r1 or c0 >= c1:
        return np.nan

    prepared = prep(geom)
    total_area = 0.0
    weighted_sum = 0.0

    for row in range(r0, r1):
        for col in range(c0, c1):
            val = raster_np[row, col]
            if np.isnan(val) or (nodata is not None and val == nodata):
                continue

            px_minx, px_maxy = transform * (col, row)
            px_maxx, px_miny = transform * (col + 1, row + 1)
            pixel = box(px_minx, px_miny, px_maxx, px_maxy)

            if not prepared.intersects(pixel):
                continue

            inter = geom.intersection(pixel)
            if not inter.is_empty:
                area = inter.area
                total_area += area
                weighted_sum += val * area

    return np.nan if total_area == 0 else weighted_sum / total_area

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


# ------------------------------
# Worker Task
# ------------------------------

@delayed
def process_polygon_block_lazy(raster_np, transform, nodata, poly_indices, year, month, temp_shp_path, unique_id):
    """Dask worker: compute zonal stats for a block of polygons."""
    gdf_slice = gpd.read_file(temp_shp_path, rows=slice(poly_indices[0], poly_indices[1]))
    polygon_block = list(zip(gdf_slice[unique_id], gdf_slice.geometry))
    
    results = []
    for uid, geom in polygon_block:
        val = area_weighted_mean_optimized(geom, raster_np, transform, nodata)
        results.append((uid, val))
    return year, month, results

@delayed
def write_month_parquet(month_results, cache_dir, variable_name, unique_id):
    """Dask worker: write monthly results to parquet with atomic file operations."""
    year, month = month_results[0][:2]
    data = [(uid, val) for _, _, block in month_results for uid, val in block]
    
    df = pd.DataFrame(data, columns=[unique_id, variable_name])
    df['year'] = year
    df['month'] = month
    
    parquet_path = get_parquet_filename(Path(cache_dir), year, month)
          
    #atomic write
    try:
        tmp_out = parquet_path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp_out, index=False)  
        tmp_out.rename(parquet_path)
        return str(parquet_path)
    except Exception as e:
        tmp_out.unlink(missing_ok=True)
        return f"Failed {year}-{month:02d}: {e}"


# ------------------------------
# Main Computation logic
# ------------------------------

def compute_zonal_statistics(polygon_shapefile, unique_id, cache_dir, output_dir, netcdf_file, variable_name="soil_moisture", block_size=1000, batch_size=12):
    """Compute monthly zonal statistics from NetCDF to parquet cache.
    
    Processes only months not already cached. Uses Dask for parallel computation.
    Returns True if new data was processed, False if all cached.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return_value = False
    
    logger.info("Opening NetCDF...")
    try:
        ds = xr.open_dataset(netcdf_file, chunks={"time": 1})
    except FileNotFoundError:
        logger.error(f"NetCDF file not found: {netcdf_file}")
        return return_value 
    except Exception as e:
        logger.error(f"Failed to open NetCDF {netcdf_file}: {e}")
        return return_value
    
    # Determine end_date from netcdf if not specified
    all_times = pd.to_datetime(ds.time.values)
    start_date = pd.to_datetime(config.START_DATE)
    end_date = pd.to_datetime(config.END_DATE) if config.END_DATE else all_times[-1]
    
    da = ds[variable_name].sel(time=slice(start_date, end_date))
    times = pd.to_datetime(da.time.values)
    
    # Generate job list excluding cached results
    existing_months = get_existing_months(cache_dir)
    job_list = [(pd.Timestamp(t).year, pd.Timestamp(t).month, i) 
                for i, t in enumerate(times) 
                if (pd.Timestamp(t).year, pd.Timestamp(t).month) not in existing_months]
    
    if not job_list:
        logger.info("No new months to process. All data cached.")
        return return_value
    
    logger.info(f"Processing {len(job_list)} months: {job_list[0][:2]} to {job_list[-1][:2]}")
    
    client = start_dask(n_workers=10)
    
    if da.rio.crs is None:
        da = da.rio.write_crs(config.CRS_FALLBACK)
    target_crs = da.rio.crs
    transform = da.rio.transform()
    nodata = da.rio.nodata

    # Pre-project polygons
    logger.info(f"Projecting shapefile to {target_crs}...")
    try:
        gdf = gpd.read_file(polygon_shapefile, columns=[unique_id, "geometry"])
        gdf = gdf.to_crs(target_crs)
    except Exception as e:
        logger.error(f"Failed to read/project shapefile {polygon_shapefile}: {e}")
        return return_value
    
    temp_dir = tempfile.mkdtemp()
    temp_shp = Path(temp_dir) / "projected_polygons.shp"
    gdf.to_file(temp_shp)
    total_polygons = len(gdf)
    del gdf

    poly_index_ranges = [(i, min(i + block_size, total_polygons)) 
                         for i in range(0, total_polygons, block_size)]
    
    # Process months in batches
    for batch_start in range(0, len(job_list), batch_size):
        batch = job_list[batch_start:batch_start + batch_size]
        logger.info(f"Batch processing {len(batch)} months: {batch[0][:2]} to {batch[-1][:2]}")
        
        # Build task graph: computation + writing
        month_tasks = {}
        for year, month, i in batch:
            month_da = da.isel(time=i).compute()
            raster_future = client.scatter(month_da.values, broadcast=True)
            
            # Computation tasks for this month
            month_tasks[(year, month)] = [
                process_polygon_block_lazy(raster_future, transform, nodata, idx_range, year, month, str(temp_shp), unique_id)
                for idx_range in poly_index_ranges
            ]
        
        # Write tasks depend on computation tasks
        write_tasks = [
            write_month_parquet(tasks, str(cache_dir), variable_name, unique_id)
            for tasks in month_tasks.values()
        ]
        
        # Execute: compute + write in parallel
        written_files = client.compute(write_tasks, sync=True)
        for filepath in written_files:
            if filepath.startswith("Failed"):
                logger.error(filepath)
            else:
                logger.info(f"Saved {filepath}")
                return_value=True

    shutil.rmtree(temp_dir)
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
    
    try:
        dfs = [pd.read_parquet(f) for f in all_parquets]
        combined = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        logger.error(f"Failed to read parquet files: {e}")
        return
        
    # Group by decade
    combined['decade'] = (combined['year'] // 10) * 10
    for decade, group in combined.groupby('decade'):
        start_year = decade
        end_year = decade + 9
        zip_path = output_dir / f"soil_moisture_{start_year}_{end_year}.zip"
        csv_name  = f"soil_moisture_{start_year}_{end_year}.csv"
        try:
            group[[unique_id, 'year', 'month', variable_name]].to_csv(
                zip_path,
                index=False,
                compression={'method': 'zip', 'archive_name': csv_name}
            )
            logger.info(f"Saved {zip_path}")
        except Exception as e:
            logger.error(f"Failed to write {zip_path}: {e}")

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