"""
AVHRR/MODIS NDVI Harmonization Pipeline (1986-Present)

Harmonizes AVHRR (1986-2013) and MODIS (2000+) NDVI for 230k ANAE wetland polygons.
Uses per-polygon Huber regression on 166-month overlap period (Mar 2000-Dec 2013).

EXECUTION FLOW:
    1. Check for missing monthly data → Submit GEE export tasks if needed
    2. Load overlap period data (2000-2013) for both sensors
    3. Calibrate AVHRR→MODIS using parallel Huber regression (ε=1.35)
    4. Save calibration parameters immediately (per-polygon + global fallback)
    5. Process all years in parallel applying harmonization
    6. Consolidate into compressed annual archive

OUTPUTS:
    NDVI_Harmonized_Annual.zip - Annual CSV files (39 years)
    calibration_params.csv - Per-polygon slope/intercept (270k polygons)
    global_params.csv - Fallback calibration parameters
    harmonization_quality.csv - R², RMSE, sample counts per polygon

SENSOR PRIORITY: MODIS direct (2000+) > AVHRR harmonized (1986-1999)
PARALLEL PROCESSING: Up to 32 cores for calibration + annual assembly

Author: Shane Brooks (brooks.eco) | Date: 2025-01-15
"""

# =============================================================================
# IMPORTS (Execution Order)
# =============================================================================

# Environment & Logging
import logging
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Parallel Processing
import multiprocessing as mp

# Data Processing
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
import zipfile

# Google Earth Engine
import ee
import gee_task_watchdog

# Initialize Environment
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths & Assets
DEFAULT_OUT_DIR = "./avhrr_modis_output"
GEE_ASSET_ID = "projects/ee-litepc/assets/ANAEv3_gt1ha"

# Date Ranges
START_DATE = (2025, 1)  # User configurable start
END_DATE = (2025, 12)  # User configurable end (set to None for auto-detect)
AVHRR_END = (2013, 12)
MODIS_START = (2000, 3)
OVERLAP_PERIOD = (2000, 3, 2013, 12)  # 166 months for calibration

# Processing Parameters
SCALE_FACTOR = 0.0001  # NDVI scaling
REDUCE_SCALE = 250  # MODIS native resolution
MIN_HARMONIZATION_SAMPLES = 24  # Minimum overlap months for per-polygon calibration
MAX_PROCESSOR_COUNT = int(os.getenv('MAX_PROCESSOR_COUNT', mp.cpu_count()))

# GEE Configuration
USE_CLOUD_STORAGE = False
GCS_BUCKET = os.getenv('GCS_BUCKET', 'your-bucket')
GEE_PROJECT = os.getenv('GEE_PROJECT', None)

# Sensor Collections
SENSOR_COLLECTIONS = {
    'AVHRR': 'NOAA/CDR/AVHRR/NDVI/V5',
    'MODIS': 'MODIS/061/MOD13Q1'
}

# Testing (Production: None)
TEST_POLYGON_LIMIT = None
USE_RANDOM_SAMPLE = False

# =============================================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# =============================================================================

def init_gee():
    """Initialize Google Earth Engine with appropriate authentication."""
    if USE_CLOUD_STORAGE:
        from google.oauth2 import service_account
        sa_json = os.getenv('GEE_SERVICE_ACCOUNT_JSON')
        creds = service_account.Credentials.from_service_account_file(
            sa_json, scopes=['https://www.googleapis.com/auth/earthengine'])
        ee.Initialize(creds)
        logger.info("✓ GEE initialized (service account)")
    else:
        try:
            ee.Initialize(project=GEE_PROJECT)
            logger.info("✓ GEE initialized (user OAuth)")
        except:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
            logger.info("✓ GEE authenticated")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_months(start_year, start_month, end_year, end_month):
    """Generate list of (year, month) tuples for date range."""
    months, y, m = [], start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m = m % 12 + 1
        y += (m == 1)
    return months

def get_processing_date_range():
    """Determine actual processing date range using latest MODIS."""
    start_year, start_month = START_DATE
    latest_year, latest_month = get_latest_modis_month()
    
    if END_DATE is None:
        end_year, end_month = latest_year, latest_month
        logger.info(f"Processing range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d} (auto-detected)")
    else:
        user_end_year, user_end_month = END_DATE
        if (user_end_year, user_end_month) <= (latest_year, latest_month):
            end_year, end_month = user_end_year, user_end_month
            logger.info(f"Processing range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d} (user-specified)")
        else:
            end_year, end_month = latest_year, latest_month
            logger.warning(f"User end date {user_end_year}-{user_end_month:02d} exceeds latest MODIS {latest_year}-{latest_month:02d}")
            logger.info(f"Processing range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d} (capped to latest MODIS)")
    
    return start_year, start_month, end_year, end_month

def get_latest_modis_month():
    """Get the date of the last image in the MODIS collection."""
    try:
        # Get the latest image date directly
        latest_image = ee.ImageCollection('MODIS/061/MOD13Q1').limit(1, 'system:time_start', False).first()
        latest_date_ms = latest_image.get('system:time_start').getInfo()
        latest_date = datetime.fromtimestamp(latest_date_ms / 1000)
        
        logger.info(f"✓ Latest MODIS: {latest_date.year}-{latest_date.month:02d}")
        return latest_date.year, latest_date.month
    except Exception as e:
        raise RuntimeError(f"Could not get latest MODIS date: {e}")

# =============================================================================
# GOOGLE EARTH ENGINE EXPORT
# =============================================================================

def export_monthly_ndvi(months_to_export, polygons):
    """
    Submit GEE export tasks for monthly NDVI extraction.
    
    Applies quality filtering:
    - AVHRR: Exclude clouds (QA bit 1) and water (QA bit 3)
    - MODIS: Retain good/marginal quality (SummaryQA bits 0-1 ≤ 1)
    - Both: Exclude NDVI < 0 (open water)
    """
    # Murray-Darling Basin AOI
    aoi = ee.Geometry.Polygon([[[138.5,-37.6], [152.5,-37.6], [152.5,-24.5], [138.5,-24.5]]], None, False)
    tasks = []
    
    for y, m in months_to_export:
        ym = f"{y:04d}{m:02d}"
        start = ee.Date.fromYMD(y, m, 1)
        end = start.advance(1, 'month')
        
        # Determine sensor availability
        sensors_to_export = []
        if y <= AVHRR_END[0] and (y < AVHRR_END[0] or m <= AVHRR_END[1]):
            sensors_to_export.append('AVHRR')
        if y >= MODIS_START[0] and (y > MODIS_START[0] or m >= MODIS_START[1]):
            sensors_to_export.append('MODIS')
        
        # Export each available sensor
        for sensor_name in sensors_to_export:
            if sensor_name == 'AVHRR':
                coll = ee.ImageCollection(SENSOR_COLLECTIONS['AVHRR']).filterBounds(aoi).filterDate(start, end)
                def mask_avhrr(img):
                    qa = img.select('QA')
                    cloud_mask = qa.bitwiseAnd(1 << 1).eq(0)  # bit 1 = 0 (not cloudy)
                    water_mask = qa.bitwiseAnd(1 << 3).eq(0)  # bit 3 = 0 (not water)
                    return img.updateMask(cloud_mask.And(water_mask))
                coll = coll.map(mask_avhrr)
                monthly_img = coll.select('NDVI').mean().multiply(SCALE_FACTOR).clip(aoi)
            else:  # MODIS
                coll = ee.ImageCollection(SENSOR_COLLECTIONS['MODIS']).filterBounds(aoi).filterDate(start, end)
                def mask_modis(img):
                    qa = img.select('SummaryQA')
                    return img.updateMask(qa.bitwiseAnd(3).lte(1))  # bits 0-1: 0 or 1
                coll = coll.map(mask_modis)
                monthly_img = coll.select('NDVI').mean().multiply(SCALE_FACTOR).clip(aoi)
            
            # Vegetation mask: NDVI >= 0 (exclude water)
            monthly_img = monthly_img.updateMask(monthly_img.gte(0))
            
            # Zonal statistics per polygon
            stats = monthly_img.reduceRegions(
                collection=polygons,
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True).combine(ee.Reducer.count(), '', True),
                scale=REDUCE_SCALE,
                tileScale=4
            ).map(lambda f: f.select(['mean', 'stdDev', 'count', 'UID'], ['NDVI', 'NDVI_sd', 'pixel_count', 'UID']).set('yearmonth', ym)
            ).filter(ee.Filter.notNull(['NDVI']))
            
            # Submit export task
            desc = f"NDVI_{sensor_name}_{ym}"
            if USE_CLOUD_STORAGE:
                task = ee.batch.Export.table.toCloudStorage(
                    collection=stats, description=desc, bucket=GCS_BUCKET,
                    fileNamePrefix=f"GEE_AVHRR_MODIS_NDVI/{desc}", fileFormat='CSV',
                    selectors=['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count'])
            else:
                task = ee.batch.Export.table.toDrive(
                    collection=stats, description=desc, folder='GEE_AVHRR_MODIS_NDVI', fileFormat='CSV',
                    selectors=['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count'])
            
            task.start()
            tasks.append(task)
            logger.info(f"✓ Task started: {desc}")
    
    logger.info("Monitor: https://code.earthengine.google.com/tasks")
    return tasks

# =============================================================================
# SENSOR HARMONIZATION (Parallel Processing)
# =============================================================================

def calibrate_uid_slice_merged(merged_slice, process_id):
    """
    Optimized UID calibration using pre-merged DataFrame slice.
    """
    params, quality = {}, {}
    uid_groups = merged_slice.groupby('UID', observed=True)
    total_uids = len(uid_groups)

    for i, (uid, group) in enumerate(uid_groups):

        if len(group) < MIN_HARMONIZATION_SAMPLES:
            continue

        X, y = group[['NDVI_src']].values, group['NDVI_tgt'].values

        try:
            model = HuberRegressor(epsilon=1.35).fit(X, y)
            y_pred = model.predict(X)
            
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(ss_res / len(y))

            params[uid] = (model.coef_[0], model.intercept_)
            quality[uid] = {'r2': r2, 'rmse': rmse, 'n': len(group)}
        except Exception:
            continue

    return params, quality

def calibrate_sensor_pair(df_source, df_target, chunk_size=50000, n_processes=None):
    """
    Optimized chunked parallel calibration for memory efficiency.
    """
    # Find common UIDs using index intersection (faster than set operations)
    common_uids = df_source['UID'].cat.categories.intersection(df_target['UID'].cat.categories).tolist()
    logger.info(f"{len(common_uids)} common UIDs to process in chunks of {chunk_size}")
    
    n_processes = n_processes or min(MAX_PROCESSOR_COUNT, mp.cpu_count())
    all_params, all_quality, global_sample_data = {}, {}, []
    
    # Pre-filter to common UIDs once
    src_filtered = df_source[df_source['UID'].isin(common_uids)][['UID', 'yearmonth', 'NDVI']]
    tgt_filtered = df_target[df_target['UID'].isin(common_uids)][['UID', 'yearmonth', 'NDVI']]
    
    # Process in chunks
    for i in range(0, len(common_uids), chunk_size):
        chunk_uids = common_uids[i:i+chunk_size]
        logger.info(f"Chunk {i//chunk_size + 1}/{(len(common_uids)-1)//chunk_size + 1}")
        
        # Chunk and merge
        src_chunk = src_filtered[src_filtered['UID'].isin(chunk_uids)]
        tgt_chunk = tgt_filtered[tgt_filtered['UID'].isin(chunk_uids)]
        merged_chunk = pd.merge(src_chunk, tgt_chunk, on=['UID', 'yearmonth'], suffixes=('_src', '_tgt'))
        
        if merged_chunk.empty:
            continue
        
        # Sample for global calibration
        if len(global_sample_data) < 50000:
            global_sample_data.append(merged_chunk.sample(min(500, len(merged_chunk))))
        
        # Split chunk for parallel processing
        chunk_slices = np.array_split(chunk_uids, n_processes)
        args_list = [(merged_chunk[merged_chunk['UID'].isin(slice_uids)], j) 
                    for j, slice_uids in enumerate(chunk_slices) if len(slice_uids) > 0]
        
        with mp.Pool(n_processes) as pool:
            results = pool.starmap(calibrate_uid_slice_merged, args_list)
        
        for params, quality in results:
            all_params.update(params)
            all_quality.update(quality)
        
        # Log chunk completion
        chunk_pct = ((i//chunk_size + 1) / ((len(common_uids)-1)//chunk_size + 1)) * 100
        logger.info(f"Chunk {i//chunk_size + 1}/{(len(common_uids)-1)//chunk_size + 1} complete ({chunk_pct:.1f}%)")
    
    # Global calibration
    global_params = (1.0, 0.0)
    if global_sample_data:
        sample = pd.concat(global_sample_data, ignore_index=True)
        model = HuberRegressor(epsilon=1.35).fit(sample[['NDVI_src']], sample['NDVI_tgt'])
        global_params = (model.coef_[0], model.intercept_)
        logger.info(f"Global slope={global_params[0]:.4f}, intercept={global_params[1]:.4f}")
    
    logger.info(f"Per-polygon: {len(all_params)}/{len(common_uids)} calibrated")
    return all_params, all_quality, global_params

def build_overlap_calibration(out_dir, processing_years):
    """
    Build AVHRR→MODIS calibration from overlap period (Mar 2000 - Dec 2013).
    
    Loads 166 months of overlap data, performs parallel calibration,
    and saves parameters immediately for resume capability.
    """
    # Check if any processing years need AVHRR harmonization
    needs_avhrr = any(year <= AVHRR_END[0] for year in processing_years)
    if not needs_avhrr:
        logger.info("✓ No AVHRR data in processing years - skipping calibration")
        return {}, {}, (1.0, 0.0)
    
    overlap_start = OVERLAP_PERIOD[0] * 100 + OVERLAP_PERIOD[1]
    overlap_end = OVERLAP_PERIOD[2] * 100 + OVERLAP_PERIOD[3]
    
    # Find overlap period files
    overlap_files = []
    for y in range(OVERLAP_PERIOD[0], OVERLAP_PERIOD[2] + 1):
        for m in range(1, 13):
            if (y * 100 + m) < overlap_start or (y * 100 + m) > overlap_end:
                continue
            overlap_files.extend(Path(out_dir).glob(f"NDVI_*_{y:04d}{m:02d}.zip"))
            overlap_files.extend(Path(out_dir).glob(f"NDVI_*_{y:04d}{m:02d}.csv"))
    
    if not overlap_files:
        logger.warning("No overlap period files found")
        return {}, {}, (1.0, 0.0)
    
    # Load overlap data with optimized dtypes
    overlap_data = []
    uid_categories = None
    
    for file_path in overlap_files:
        try:
            sensor = file_path.stem.split('_')[1]
            df = pd.read_csv(file_path, dtype={'UID': 'category', 'yearmonth': 'int32'})
            if 'yearmonth' in df.columns and not df.empty:
                df['sensor'] = sensor
                # Ensure consistent UID categories across files
                if uid_categories is None:
                    uid_categories = df['UID'].cat.categories
                else:
                    df['UID'] = df['UID'].cat.set_categories(uid_categories, ordered=False)
                overlap_data.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file_path.name}: {e}")
    
    if not overlap_data:
        return {}, {}, (1.0, 0.0)
    
    df_overlap = pd.concat(overlap_data, ignore_index=True)
    # UID already categorical, yearmonth already int32
    
    # Verify both sensors present
    has_avhrr = (df_overlap['sensor'] == 'AVHRR').any()
    has_modis = (df_overlap['sensor'] == 'MODIS').any()
    
    if has_avhrr and has_modis:
        logger.info("Calibrating AVHRR→MODIS from overlap period...")
        uids = df_overlap['UID'].unique()
        params, quality, global_params = calibrate_sensor_pair(
            df_overlap[df_overlap['sensor'] == 'AVHRR'],
            df_overlap[df_overlap['sensor'] == 'MODIS'])
        
        # Save parameters immediately (resume capability)
        if params:
            params_df = pd.DataFrame([(uid, slope, intercept) for uid, (slope, intercept) in params.items()], 
                                    columns=['UID', 'slope', 'intercept'])
            params_df.to_csv(Path(out_dir) / "calibration_params.csv", index=False)
            logger.info(f"✓ Saved {len(params)} per-polygon calibration parameters")
        
        global_df = pd.DataFrame([{'slope': global_params[0], 'intercept': global_params[1]}])
        global_df.to_csv(Path(out_dir) / "global_params.csv", index=False)
        logger.info(f"✓ Saved global parameters: slope={global_params[0]:.4f}, intercept={global_params[1]:.4f}")
        
        return params, quality, global_params
    else:
        logger.info("No overlap data - using identity transform")
        return {}, {}, (1.0, 0.0)

# =============================================================================
# ANNUAL DATA PROCESSING (Parallel by Year)
# =============================================================================

def process_year_data(year, out_dir, params, global_params):
    """
    Optimized single year processing with sensor harmonization.
    """
    year_files = list(Path(out_dir).glob(f"NDVI_*_{year}*.zip")) + list(Path(out_dir).glob(f"NDVI_*_{year}*.csv"))
    if not year_files:
        return None
    
    # Load and concatenate in one step
    year_data = []
    for file_path in year_files:
        try:
            df = pd.read_csv(file_path, dtype={'UID': 'category', 'yearmonth': 'int32'})
            if not df.empty and 'yearmonth' in df.columns:
                df['sensor'] = file_path.stem.split('_')[1]
                year_data.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file_path.name}: {e}")
    
    if not year_data:
        return None
    
    df_year = pd.concat(year_data, ignore_index=True)
    # UID already categorical, yearmonth already int32
    
    # Optimized sensor selection and harmonization
    df_year = df_year.drop_duplicates(['UID', 'yearmonth', 'sensor'])
    
    # Split by sensor
    modis_mask = df_year['sensor'] == 'MODIS'
    modis_data = df_year[modis_mask].copy()
    avhrr_data = df_year[~modis_mask].copy()
    
    # Filter AVHRR to non-overlapping observations
    if not modis_data.empty and not avhrr_data.empty:
        modis_keys = pd.MultiIndex.from_arrays([modis_data['UID'], modis_data['yearmonth']])
        avhrr_keys = pd.MultiIndex.from_arrays([avhrr_data['UID'], avhrr_data['yearmonth']])
        avhrr_filtered = avhrr_data[~avhrr_keys.isin(modis_keys)].copy()
    else:
        avhrr_filtered = avhrr_data.copy()
    
    modis_data['sensor_source'] = 'MODIS'
    
    # Vectorized AVHRR harmonization
    if not avhrr_filtered.empty and params:
        # Vectorized calibration using map for better performance
        avhrr_filtered['slope'] = avhrr_filtered['UID'].map(lambda uid: params.get(uid, global_params)[0])
        avhrr_filtered['intercept'] = avhrr_filtered['UID'].map(lambda uid: params.get(uid, global_params)[1])
        
        # Vectorized harmonization
        avhrr_filtered['NDVI'] = avhrr_filtered['NDVI'] * avhrr_filtered['slope'] + avhrr_filtered['intercept']
        avhrr_filtered['sensor_source'] = 'AVHRR_harmonized'
        
        # Drop temporary columns to save memory
        avhrr_harmonized = avhrr_filtered.drop(['slope', 'intercept'], axis=1)
    else:
        avhrr_harmonized = pd.DataFrame()
    
    # Combine results
    df_harmonized = pd.concat([modis_data, avhrr_harmonized], ignore_index=True)
    
    # Calculate sensor counts
    sensor_counts = {'MODIS': len(modis_data), 'AVHRR': len(avhrr_harmonized)}
    
    if not df_harmonized.empty:
        df_harmonized['NDVI'] = df_harmonized['NDVI'].round(4)
        df_harmonized['NDVI_sd'] = df_harmonized['NDVI_sd'].round(4)
        df_harmonized = df_harmonized[['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count', 'sensor_source']]
        
        # Save as CSV file
        csv_path = Path(out_dir) / f"NDVI_Harmonized_{year}.csv"
        df_harmonized.to_csv(csv_path, index=False)
        
        return year, len(df_harmonized), sensor_counts
    
    return None

def create_decadal_zips(out_dir, processed_years):
    """
    Create decadal zip files from annual CSV files.
    Groups: 1986-1995, 1996-2005, 2006-2015, 2016-2025
    """
    decades = [
        (1986, 1995), (1996, 2005), (2006, 2015), (2016, 2025)
    ]
    
    for start_year, end_year in decades:
        decade_years = [y for y in processed_years if start_year <= y <= end_year]
        if not decade_years:
            continue
            
        # Determine actual year range for filename
        actual_start = min(decade_years)
        actual_end = max(decade_years)
        
        zip_name = f"NDVI_{actual_start}-{actual_end}_ANAEv3_monthly_AVHRR-MODIS.zip"
        zip_path = Path(out_dir) / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for year in sorted(decade_years):
                csv_file = Path(out_dir) / f"NDVI_Harmonized_{year}.csv"
                if csv_file.exists():
                    zf.write(csv_file, f"NDVI_Harmonized_{year}.csv")
                    csv_file.unlink()  # Remove CSV after adding to zip
        
        logger.info(f"✓ Created {zip_name} with {len(decade_years)} years")

def process_annual_data(out_dir, start_year=None, start_month=None, end_year=None, end_month=None):
    """
    Parallel processing of annual harmonized data.
    
    Each year processed independently to eliminate file conflicts.
    Results consolidated into compressed annual archive.
    """
    logger.info("Processing annual data in parallel by year...")
    
    # Find all available years efficiently (exclude ANAEv3 files)
    all_files = list(Path(out_dir).glob("NDVI_*_*.zip")) + list(Path(out_dir).glob("NDVI_*_*.csv"))
    all_files = [f for f in all_files if "ANAEv3" not in f.name]
    years = set()
    overlap_files_exist = False
    
    # Use passed date range or fallback to config
    if start_year is None:
        start_year, start_month = START_DATE
    if end_year is None:
        end_year, end_month = END_DATE if END_DATE else (9999, 12)
    
    processing_start = start_year * 100 + start_month
    processing_end = end_year * 100 + end_month
    
    for file_path in all_files:
        try:
            ym = file_path.stem.split('_')[2]
            year = int(ym[:4])
            month = int(ym[4:6])
            ym_int = year * 100 + month
            
            # Only process files within start_date and end_date
            if not (processing_start <= ym_int <= processing_end):
                continue
                
            years.add(year)
            
            # Check if this file is within overlap period
            overlap_start = OVERLAP_PERIOD[0] * 100 + OVERLAP_PERIOD[1]
            overlap_end = OVERLAP_PERIOD[2] * 100 + OVERLAP_PERIOD[3]
            if overlap_start <= ym_int <= overlap_end:
                overlap_files_exist = True
        except:
            continue
    
    if not years:
        logger.warning("No data files found within processing date range")
        return
    
    # Get calibration parameters (only if overlap files exist and AVHRR years need processing)
    needs_avhrr = any(year <= AVHRR_END[0] for year in years)
    if needs_avhrr and overlap_files_exist:
        params, quality, global_params = build_overlap_calibration(out_dir, years)
    else:
        params, quality, global_params = {}, {}, (1.0, 0.0)
        if needs_avhrr:
            logger.info("✓ No overlap period files found - skipping calibration")
        else:
            logger.info("✓ No AVHRR data in processing years - skipping calibration")
    
    # Process years in parallel
    n_processes = min(MAX_PROCESSOR_COUNT, mp.cpu_count(), len(years))
    logger.info(f"Processing {len(years)} years across {n_processes} processes...")
    
    with mp.Pool(n_processes) as pool:
        args = [(year, out_dir, params, global_params) for year in sorted(years)]
        results = pool.starmap(process_year_data, args)
    
    # Collect results (no consolidation needed)
    total_sensor_counts = {'MODIS': 0, 'AVHRR': 0}
    processed_years = []
    
    for result in results:
        if result:
            year, obs_count, sensor_counts = result
            total_sensor_counts['MODIS'] += sensor_counts['MODIS']
            total_sensor_counts['AVHRR'] += sensor_counts['AVHRR']
            processed_years.append(year)
            logger.info(f"  ✓ Year {year}: {obs_count} observations → NDVI_Harmonized_{year}.csv")
    
    # Create decadal zip files
    create_decadal_zips(out_dir, processed_years)
    
    # Save quality control outputs
    if quality:
        pd.DataFrame(quality).T.to_csv(Path(out_dir) / "harmonization_quality.csv")
    
    # Final summary
    logger.info(f"✓ Processed {len(processed_years)} years")
    logger.info(f"✓ Selected: MODIS={total_sensor_counts['MODIS']}, AVHRR={total_sensor_counts['AVHRR']}")
    logger.info(f"✓ Harmonization complete: {len(params)} per-polygon calibrations")
    logger.info(f"✓ Decadal zip files created")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution pipeline:
    1. Check for missing monthly data
    2. Submit GEE exports if needed OR process existing data
    3. Perform harmonization and generate annual outputs
    """
    out_dir = Path(DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GEE for auto-detection
    init_gee()
    
    # Determine target date range (always check latest MODIS)
    start_year, start_month, end_year, end_month = get_processing_date_range()
    
    # Check for missing monthly data
    months = list_months(start_year, start_month, end_year, end_month)
    missing_months = []
    for y, m in months:
        ym = f"{y:04d}{m:02d}"
        month_files = list(out_dir.glob(f"NDVI_*_{ym}.csv")) + list(out_dir.glob(f"NDVI_*_{ym}.zip"))
        if not month_files:
            missing_months.append((y, m))
    
    if not missing_months:
        # All data available - proceed with harmonization
        logger.info("✓ All requested months have data - processing...")
        process_annual_data(out_dir, start_year, start_month, end_year, end_month)
        logger.info("\n" + "="*60)
        logger.info("✓ PROCESSING COMPLETE")
        logger.info(f"Output: {out_dir}")
        logger.info("="*60)
        return
    
    # Missing data - submit GEE export tasks (GEE already initialized)
    logger.info(f"Missing {len(missing_months)} months - submitting GEE export tasks...")
    
    # Configure polygon collection
    polygons = ee.FeatureCollection(GEE_ASSET_ID)
    if TEST_POLYGON_LIMIT:
        if USE_RANDOM_SAMPLE:
            logger.info(f"Selecting {TEST_POLYGON_LIMIT} random polygons...")
            polygons = polygons.randomColumn('random', 42).sort('random').limit(TEST_POLYGON_LIMIT)
            logger.info("✓ Random sample ready")
        else:
            polygons = ee.FeatureCollection(polygons.toList(TEST_POLYGON_LIMIT))
            logger.info(f"✓ Using first {TEST_POLYGON_LIMIT} polygons")
    else:
        logger.info(f"✓ Using all polygons from {GEE_ASSET_ID}")
    
    logger.info(f"Date range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    logger.info(f"Exporting {len(missing_months)} missing months")
    
    # Launch task monitoring
    if gee_task_watchdog.launch_watchdog():
        logger.info("✓ Task watchdog launched in new window")
    
    # Submit export tasks
    tasks = export_monthly_ndvi(missing_months, polygons)
    
    # Instructions for next steps
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS:")
    logger.info("1. Monitor tasks: https://code.earthengine.google.com/tasks")
    logger.info("2. Download CSVs from Google Drive: GEE_AVHRR_MODIS_NDVI")
    logger.info(f"3. Save CSVs to: {out_dir.absolute()}")
    logger.info("4. Re-run this script to harmonize")
    logger.info("="*60)

if __name__ == "__main__":
    main()