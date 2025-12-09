"""
Landsat Monthly NDVI Extraction (1986-Present) - Automated Workflow

PURPOSE:
Extract monthly NDVI from Landsat 5/7/8/9 for ANY SIZE polygon set using GEE Export tasks.
Outputs annual CSV files. Supports incremental updates (fully automated).

ARCHITECTURE:
- Computation stays on GEE servers (no memory limits)
- Results exported to Google Drive (or Cloud Storage)
- Python automatically downloads and processes CSVs
- Supports 250,000+ polygons

USAGE:
    1. First run: python ndvi-landsat-monthly.py
       → Creates GEE export tasks
       → Monitor: https://code.earthengine.google.com/tasks
    
    2. After tasks complete: python ndvi-landsat-monthly.py
       → Automatically downloads CSVs from Drive
       → Processes and harmonizes data
       → Outputs ready-to-use files
    
    3. Subsequent runs: Incremental updates (only new months)

CONFIGURATION:
    GEE_ASSET_ID: GEE asset path with 'UID' field
    TEST_MODE: True (10 polygons) or False (all polygons)
    USE_CLOUD_STORAGE: False (Drive) or True (GCS bucket)

OUTPUT FILES (in ./landsat_ndvi_output/):
    NDVI_Landsat_{year}.csv - Raw monthly NDVI by sensor
    NDVI_harmonized_{year}.csv - Cross-calibrated timeseries
    harmonization_L5_L7_quality.csv - Calibration quality metrics
    sensor_summary.csv - Data availability by sensor
    temporal_completeness.csv - Data gaps per polygon
    anomalous_jumps.csv - Quality flags (>30% month-to-month change)

SENSORS & HARMONIZATION:
    L5 (1986-2012) → L5→L7→L8 chain
    L7 pre-SLC (1999-2003) → L7→L8
    L7 post-SLC (2003+) → EXCLUDED (poor quality)
    L8/L9 (2013+) → Reference (no adjustment)

REFERENCE:
    Roy et al. (2016) RSE 185:57-70 (L7→L8 coefficients)

Author: Shane Brooks (brooks.eco)
Modified: 2025-01-08
"""

import ee
import ee.ee_exception
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from pathlib import Path
import logging
import time
import io
from datetime import datetime
from dotenv import load_dotenv
import os

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_OUT_DIR = "./landsat_ndvi_output"

# Export destination
USE_CLOUD_STORAGE = False  # False=Google Drive (user OAuth), True=Cloud Storage (service account)
GCS_BUCKET = os.getenv('GCS_BUCKET', 'your-gcs-bucket-name')
GEE_PROJECT = os.getenv('GEE_PROJECT', None)  # Optional: Cloud project ID with EE enabled

# Landsat archive dates
LANDSAT_START_YEAR = 1986  # L5 launch year, used when START_DATE = None

# Harmonization periods (scientifically defined, not just operational overlap)
L5_L7_OVERLAP = (1999, 6, 2003, 4)  # June 1999 - April 2003: Pre-SLC failure, high quality L7 data
L7_SLC_FAILURE_DATE = 200305  # May 2003: Scan Line Corrector failure (striped images)
L7_EXCLUDE_POST_SLC = True  # Exclude L7 post-May 2003 due to poor quality

# Landsat Collection 2 Level-2 scaling (converts DN to surface reflectance)
SCALE_FACTOR = 0.0000275  # Multiplicative factor from USGS documentation
OFFSET = -0.2  # Additive offset from USGS documentation

# Spatial processing
REDUCE_SCALE = 30  # Landsat pixel size in meters (30m resolution)

# Harmonization quality control
MIN_HARMONIZATION_SAMPLES = 12  # Minimum paired observations needed for robust L5→L7 calibration

# Date range configuration
# START_DATE: (year, month) tuple or None
#   - None = full archive from 1986
#   - (2025, 10) = start from October 2025
START_DATE = (2025, 11)

# END_DATE: (year, month) tuple or None  
#   - None = auto-detect latest complete month (checks L8/L9 for recent data)
#   - (2025, 11) = end at November 2025
END_DATE = (2025, 11)

# TEST_POLYGON_LIMIT: Integer or None
#   - None = process all polygons in asset
#   - 10 = limit to first 10 polygons (for testing)
TEST_POLYGON_LIMIT = None



LANDSAT_COLLECTIONS = {
    'L5': 'LANDSAT/LT05/C02/T1_L2',
    'L7': 'LANDSAT/LE07/C02/T1_L2',
    'L8': 'LANDSAT/LC08/C02/T1_L2',
    'L9': 'LANDSAT/LC09/C02/T1_L2'
}

# ============================================================================
# INITIALIZE GOOGLE EARTH ENGINE
# ============================================================================
if USE_CLOUD_STORAGE:
    # Service account authentication (for Cloud Storage)
    service_account_json = os.getenv('GEE_SERVICE_ACCOUNT_JSON')
    if not service_account_json:
        raise ValueError("USE_CLOUD_STORAGE=True requires GEE_SERVICE_ACCOUNT_JSON in .env")
    if not GOOGLE_API_AVAILABLE:
        raise ImportError("Service account requires: pip install google-auth google-api-python-client")
    if not Path(service_account_json).exists():
        raise FileNotFoundError(f"Service account file not found: {service_account_json}")
    
    credentials = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=['https://www.googleapis.com/auth/earthengine']
    )
    ee.Initialize(credentials)
    logger.info("✓ GEE initialized with service account (Cloud Storage mode)")
else:
    # User OAuth authentication (for Google Drive)
    try:
        ee.Initialize(project=GEE_PROJECT)
        logger.info("✓ GEE initialized with user OAuth (Google Drive mode)")
    except ee.EEException as e:
        if "Not signed up" in str(e):
            logger.error("✗ Earth Engine access not enabled")
            logger.error("Register at: https://signup.earthengine.google.com/")
            raise
        logger.warning("⚠ No Earth Engine credentials found")
        logger.info("Attempting authentication...")
        try:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
            logger.info("✓ Authentication successful")
        except Exception as auth_error:
            logger.error(f"✗ Authentication failed: {auth_error}")
            logger.error("Run manually: earthengine authenticate")
            raise

def load_polygons(gee_asset_id: str, limit: int = None) -> ee.FeatureCollection:
    """Load polygons from GEE asset."""
    fc = ee.FeatureCollection(gee_asset_id)
    if limit:
        fc = ee.FeatureCollection(fc.toList(limit))
        logger.info(f"✓ Using GEE asset: {gee_asset_id} (limited to {limit} polygons)")
    else:
        logger.info(f"✓ Using GEE asset: {gee_asset_id}")
    return fc

def list_months_range(start_year: int, start_month: int, end_year: int, end_month: int):
    """Return list of (year, month) inclusive from start_year/start_month to end_year/end_month."""
    months = []
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        months.append((y, m))
        # increment month
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months

def month_start_end(year: int, month: int):
    """Return ee.Date start and end for a calendar month."""
    start = ee.Date.fromYMD(year, month, 1)
    if month == 12:
        end = ee.Date.fromYMD(year + 1, 1, 1)
    else:
        end = ee.Date.fromYMD(year, month + 1, 1)
    return start, end

def get_landsat_latest_complete_month():
    now = datetime.now()
    for months_back in range(1, 4):
        check_date = now.replace(day=1)
        for _ in range(months_back):
            check_date = (check_date.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
        
        year, month = check_date.year, check_date.month
        start = ee.Date.fromYMD(year, month, 1)
        end = ee.Date.fromYMD(year + int(month==12), month % 12 + 1, 1)
        
        for cid in ['LANDSAT/LC09/C02/T1_L2', 'LANDSAT/LC08/C02/T1_L2']:
            try:
                if ee.ImageCollection(cid).filterDate(start, end).size().getInfo() > 0:
                    logger.info(f"✓ Latest Landsat month: {year}-{month:02d}")
                    return year, month
            except:
                pass
    
    return now.year, now.month

def apply_scale_factors(image):
    """Convert Landsat Collection 2 DN to surface reflectance.
    
    Landsat C2 L2 products store surface reflectance as scaled integers.
    Formula: Reflectance = DN * 0.0000275 - 0.2
    Valid range: -0.2 to 1.6 (though typical vegetation is 0.0 to 0.9)
    """
    optical = image.select('SR_B.*').multiply(SCALE_FACTOR).add(OFFSET)
    return image.addBands(optical, None, True)

def mask_clouds_landsat(image):
    """Mask clouds, shadows, and bad pixels using QA_PIXEL band.
    
    Landsat Collection 2 QA_PIXEL bit flags (16-bit integer):
    - Bit 0: Fill (no data)
    - Bit 1: Dilated cloud
    - Bit 2: Cirrus (high confidence)
    - Bit 3: Cloud (high confidence)
    - Bit 4: Cloud shadow (high confidence)
    - Bit 5: Snow
    - Bit 6: Clear
    - Bit 7: Water
    
    We mask: Bit 3 (cloud), Bit 4 (cloud shadow), Bit 5 (snow)
    L7 also masks: Bit 0 (fill values from SLC failure)
    
    Reference: https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
    """
    qa = image.select('QA_PIXEL')
    
    # Mask clouds, shadows, and snow (bits 3, 4, 5)
    # bitwiseAnd(1 << 3) extracts bit 3, .eq(0) checks if it's NOT set
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(  # Bit 3: Cloud
           qa.bitwiseAnd(1 << 4).eq(0)).And( # Bit 4: Cloud shadow
           qa.bitwiseAnd(1 << 5).eq(0))      # Bit 5: Snow
    
    # L7-specific: Also mask fill values (SLC-off striping)
    spacecraft = ee.String(image.get('SPACECRAFT_ID'))
    mask = ee.Algorithms.If(
        spacecraft.equals('LANDSAT_7'),
        mask.And(qa.bitwiseAnd(1 << 0).eq(0)),  # Bit 0: Fill
        mask
    )
    
    return image.updateMask(ee.Image(mask))

def compute_ndvi_landsat(image):
    """Compute NDVI for Landsat 4–9 with cloud masking.
    
    Processing steps:
    1. Scale DN to surface reflectance (0.0000275 * DN - 0.2)
    2. Mask clouds, shadows, snow using QA_PIXEL band
    3. Select correct NIR and Red bands (varies by sensor)
    4. Calculate NDVI = (NIR - Red) / (NIR + Red)
    
    Band mapping:
    - L4/L5/L7 (TM/ETM+): NIR=B4, Red=B3
    - L8/L9 (OLI): NIR=B5, Red=B4
    
    Returns:
        Masked NDVI image with only clear pixels (-1 to +1 range)
    """
    # Step 1: Convert to surface reflectance
    image = apply_scale_factors(image)
    
    # Step 2: Mask bad pixels (clouds, shadows, snow)
    image = mask_clouds_landsat(image)

    spacecraft = ee.String(image.get('SPACECRAFT_ID'))

    # Step 3: Select NIR band (sensor-specific)
    # L8/L9 use Band 5 (NIR), L4/L5/L7 use Band 4 (NIR)
    nir = ee.Image(
        ee.Algorithms.If(spacecraft.equals('LANDSAT_8'),
                         image.select('SR_B5'),
                         ee.Algorithms.If(spacecraft.equals('LANDSAT_9'),
                                          image.select('SR_B5'),
                                          image.select('SR_B4')))
    )

    # Step 4: Select Red band (sensor-specific)
    # L8/L9 use Band 4 (Red), L4/L5/L7 use Band 3 (Red)
    red = ee.Image(
        ee.Algorithms.If(spacecraft.equals('LANDSAT_8'),
                         image.select('SR_B4'),
                         ee.Algorithms.If(spacecraft.equals('LANDSAT_9'),
                                          image.select('SR_B4'),
                                          image.select('SR_B3')))
    )

    # Compute NDVI
    ndvi = nir.subtract(red).divide(nir.add(red).add(1e-10)).rename('NDVI')

    # Keep properties
    return ndvi.copyProperties(image, ['system:time_start', 'SPACECRAFT_ID']).set('sensor', spacecraft)



def build_monthly_collection(aoi, start_year: int, start_month: int, end_year: int, end_month: int):
    """Build monthly median images for each month in the given inclusive range."""
    collections = [ee.ImageCollection(cid) for cid in LANDSAT_COLLECTIONS.values()]
    merged = collections[0]
    for coll in collections[1:]:
        merged = merged.merge(coll)
    merged = merged.map(compute_ndvi_landsat)

    months = list_months_range(start_year, start_month, end_year, end_month)

    imgs = []
    for y, m in months:
        start, end = month_start_end(y, m)
        monthly = merged.filterDate(start, end).median().clip(aoi)
        ym = f"{y:04d}{m:02d}"
        # Set properties: yearmonth and a sensible sensor flag (COMBINED)
        monthly = monthly.set('yearmonth', ym).set('sensor', 'COMBINED').set('month_start', start.millis())
        imgs.append(monthly)
    return ee.ImageCollection(imgs)

def reduce_regions_monthly(img_collection, polygons):
    def reducer_func(img):
        stats = img.reduceRegions(
            collection=polygons,
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True).combine(ee.Reducer.count(), '', True),
            scale=REDUCE_SCALE,
            tileScale=4
        ).map(lambda f: f.select(
            ['mean','stdDev','count','UID'], 
            ['NDVI','NDVI_sd','pixel_count','UID']
        ).set({
            'yearmonth': img.get('yearmonth'),
            'date_start': img.get('system:time_start'),
            'sensor': img.get('sensor')
        }))
        return stats.filter(ee.Filter.notNull(['NDVI']))
    
    return img_collection.map(reducer_func).flatten()

def get_last_extracted_month(out_dir: str):
    csv_files = list(Path(out_dir).glob("NDVI_Landsat_*.csv"))
    if not csv_files:
        return None
    max_ym = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty and 'yearmonth' in df.columns:
                max_ym = max(max_ym, df['yearmonth'].max())
        except:
            pass
    return max_ym if max_ym > 0 else None

def build_incremental_collection(aoi, out_dir: str, start_year: int, start_month: int, end_year: int, end_month: int):
    """Build collection for months not yet extracted."""
    requested_months = list_months_range(start_year, start_month, end_year, end_month)
    
    # Check which months already exist in processed files
    csv_files = list(Path(out_dir).glob("NDVI_Landsat_*.csv"))
    existing_months = set()
    if csv_files:
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if 'yearmonth' in df.columns:
                    existing_months.update(df['yearmonth'].unique())
            except:
                pass
    
    # Filter to only missing months
    months = [(y, m) for y, m in requested_months if (y * 100 + m) not in existing_months]
    
    if existing_months:
        logger.info(f"Incremental: {len(months)} new months (already have {len(existing_months)} months)")
    else:
        logger.info(f"Full extraction: {len(months)} months")

    if not months:
        logger.info("No new months to extract")
        return None

    collections = [ee.ImageCollection(cid) for cid in LANDSAT_COLLECTIONS.values()]
    merged = collections[0]
    for coll in collections[1:]:
        merged = merged.merge(coll)
    merged = merged.map(compute_ndvi_landsat)

    imgs = []
    for y, m in months:
        start, end = month_start_end(y, m)
        monthly = merged.filterDate(start, end).median().clip(aoi)
        ym = f"{y:04d}{m:02d}"
        monthly = monthly.set('yearmonth', ym).set('sensor', 'COMBINED')
        imgs.append(monthly)
    return ee.ImageCollection(imgs)

def get_sensor_date_ranges():
    """Get operational date ranges from GEE metadata."""
    ranges = {}
    for sensor_name, collection_id in LANDSAT_COLLECTIONS.items():
        try:
            coll = ee.ImageCollection(collection_id)
            dates = coll.aggregate_array('system:time_start').getInfo()
            if dates:
                min_date = datetime.fromtimestamp(min(dates) / 1000)
                max_date = datetime.fromtimestamp(max(dates) / 1000)
                ranges[sensor_name] = {
                    'start': (min_date.year, min_date.month),
                    'end': (max_date.year, max_date.month)
                }
        except:
            pass
    return ranges

def export_monthly_tasks(start_year, start_month, end_year, end_month, aoi, polygons):
    """Export GEE tasks with smart sensor grouping.
    
    Export strategy by date:
    - 1999-2003 (L5/L7 overlap): Separate exports for harmonization calibration
    - 1986-2013 (Pre-L8): Separate L5 and L7 exports
    - 2013+ (L8/L9 era): Combined L8+L9 export for better cloud-free coverage
    
    Why combine L8+L9?
    - Doubles temporal resolution (16-day → 8-day revisit)
    - More cloud-free observations per month
    - No harmonization needed (sensors are identical)
    
    Returns:
        List of started GEE export tasks
    """
    months = list_months_range(start_year, start_month, end_year, end_month)
    
    # Define harmonization period (June 1999 - April 2003)
    l5_l7_overlap_start = L5_L7_OVERLAP[0] * 100 + L5_L7_OVERLAP[1]  # 199906
    l5_l7_overlap_end = L5_L7_OVERLAP[2] * 100 + L5_L7_OVERLAP[3]    # 200304
    
    tasks = []
    for y, m in months:
        ym = f"{y:04d}{m:02d}"
        ym_int = y * 100 + m
        start, end = month_start_end(y, m)
        
        # Determine export strategy based on date
        if ym_int >= l5_l7_overlap_start and ym_int <= l5_l7_overlap_end:
            # L5/L7 overlap period: export separately for harmonization
            sensors_to_export = [('L5', 'LANDSAT/LT05/C02/T1_L2'), 
                                ('L7', 'LANDSAT/LE07/C02/T1_L2')]
            logger.info(f"{ym}: L5/L7 overlap - exporting separately")
        elif ym_int < 201304:  # Before L8 launch (April 2013)
            # Pre-L8: export L5 and L7 separately
            sensors_to_export = [('L5', 'LANDSAT/LT05/C02/T1_L2'),
                                ('L7', 'LANDSAT/LE07/C02/T1_L2')]
        else:
            # L8/L9 era: combine for better median
            sensors_to_export = [('L8L9', None)]  # Special flag for combined
        
        for sensor_name, collection_id in sensors_to_export:
            if sensor_name == 'L8L9':
                # Combined L8+L9 export (2013+)
                # Each sensor is cloud-masked BEFORE merging
                l8_coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').map(compute_ndvi_landsat)
                l9_coll = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').map(compute_ndvi_landsat)
                merged = l8_coll.merge(l9_coll)  # Merge masked collections
                monthly_img = merged.filterDate(start, end).median().clip(aoi)  # Median of clear pixels
                desc = f"NDVI_L8L9_{ym}"
            else:
                # Individual sensor export (pre-2013 or harmonization period)
                coll = ee.ImageCollection(collection_id).map(compute_ndvi_landsat)
                monthly_img = coll.filterDate(start, end).median().clip(aoi)
                desc = f"NDVI_{sensor_name}_{ym}"
            
            # Note: We don't check if data exists here to avoid expensive getInfo() calls
            # GEE will handle empty collections gracefully during export
            # If no data exists, the export will complete but produce an empty CSV
            
            # Compute zonal statistics per polygon
            # - mean: Average NDVI across polygon
            # - stdDev: Within-polygon variability (vegetation patchiness)
            # - count: Number of clear 30m pixels (quality indicator)
            stats = monthly_img.reduceRegions(
                collection=polygons,
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True).combine(ee.Reducer.count(), '', True),
                scale=REDUCE_SCALE,  # 30m Landsat resolution
                tileScale=4  # Process in 4x4 tile chunks (memory optimization)
            ).map(lambda f: f.select(
                ['mean', 'stdDev', 'count', 'UID'],
                ['NDVI', 'NDVI_sd', 'pixel_count', 'UID']
            ).set('yearmonth', ym)).filter(ee.Filter.notNull(['NDVI']))  # Remove polygons with no clear pixels

            if USE_CLOUD_STORAGE:
                task = ee.batch.Export.table.toCloudStorage(
                    collection=stats,
                    description=desc,
                    bucket=GCS_BUCKET,
                    fileNamePrefix=f"GEE_Landsat_NDVI/{desc}",
                    fileFormat='CSV',
                    selectors=['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count']
                )
            else:
                task = ee.batch.Export.table.toDrive(
                    collection=stats,
                    description=desc,
                    folder='GEE_Landsat_NDVI',
                    fileFormat='CSV',
                    selectors=['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count']
                )
            task.start()
            tasks.append(task)
            logger.info(f"✓ Task started: {desc}")

    logger.info(f"Monitor at: https://code.earthengine.google.com/tasks")
    return tasks

def export_to_drive(fc, description, out_dir):
    """Export FeatureCollection to Google Drive or Cloud Storage."""
    if USE_CLOUD_STORAGE:
        task = ee.batch.Export.table.toCloudStorage(
            collection=fc,
            description=description,
            bucket=GCS_BUCKET,
            fileNamePrefix=f"GEE_Landsat_NDVI/{description}",
            fileFormat='CSV'
        )
    else:
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=description,
            folder='GEE_Landsat_NDVI',
            fileFormat='CSV'
        )
    task.start()
    logger.info(f"✓ Export task started: {description}")
    logger.info(f"  Monitor at: https://code.earthengine.google.com/tasks")
    return task

def wait_for_tasks(tasks, check_interval=60):
    """Wait for GEE tasks to complete."""
    logger.info(f"Waiting for {len(tasks)} export tasks to complete...")
    logger.info("This may take 10-60 minutes depending on polygon count and date range")
    
    while True:
        states = [task.status()['state'] for task in tasks]
        completed = states.count('COMPLETED')
        failed = states.count('FAILED')
        running = states.count('RUNNING')
        
        if completed + failed == len(tasks):
            logger.info(f"✓ All tasks finished: {completed} completed, {failed} failed")
            return completed, failed
        
        logger.info(f"Status: {completed} completed, {running} running, {failed} failed")
        time.sleep(check_interval)

def download_from_drive(out_dir):
    """Check for downloaded CSVs from Google Drive.
    
    Returns:
        True if CSV files exist, False if no files found
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Check if files already downloaded
    csv_files = list(out_path.glob("NDVI_*.csv"))
    if csv_files:
        logger.info(f"✓ Found {len(csv_files)} CSV files in {out_dir}")
        return True
    
    # No files found - inform user but don't block
    logger.info("ℹ No CSV files found locally")
    logger.info("  (First run or files not yet downloaded from Drive)")
    return False

def process_downloaded_csvs(out_dir):
    """Process CSVs downloaded from Google Drive."""
    out_dir = Path(out_dir)
    csv_files = list(out_dir.glob("NDVI_L*_*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {out_dir}")
        return False
    
    logger.info(f"✓ Found {len(csv_files)} CSV files")
    
    # Organize by year, adding sensor from filename
    for csv_file in csv_files:
        try:
            # Extract sensor from filename: NDVI_L5_YYYYMM.csv -> L5, NDVI_L8L9_YYYYMM.csv -> L8L9
            parts = csv_file.stem.split('_')
            if len(parts) < 3:
                continue
            sensor = parts[1]  # L5, L7, L8L9
            
            df = pd.read_csv(csv_file)
            if 'yearmonth' not in df.columns:
                continue
            
            # Add sensor column from filename
            if sensor == 'L8L9':
                df['sensor'] = 'LANDSAT_8_9'  # Combined sensor
            else:
                df['sensor'] = f"LANDSAT_{sensor[1]}"  # L5 -> LANDSAT_5
            
            for year in df['yearmonth'].unique() // 100:
                df_year = df[df['yearmonth'] // 100 == year]
                out_file = out_dir / f"NDVI_Landsat_{year}.csv"
                
                if out_file.exists():
                    df_existing = pd.read_csv(out_file)
                    df_combined = pd.concat([df_existing, df_year], ignore_index=True)
                    df_combined.drop_duplicates(subset=['UID','yearmonth','sensor'], inplace=True)
                else:
                    df_combined = df_year
                
                df_combined.to_csv(out_file, index=False)
            
            logger.info(f"✓ Processed: {csv_file.name}")
        except Exception as e:
            logger.warning(f"Error processing {csv_file.name}: {e}")
    
    return True

def harmonize_landsat_sensors(local_out_dir: str):
    """Harmonize L5/L7 to L8/L9 scale."""
    logger.info("Starting sensor harmonization...")
    csv_files = list(Path(local_out_dir).glob("NDVI_Landsat_*.csv"))
    if not csv_files:
        logger.warning("No CSV files found for harmonization")
        return
    
    df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df_all['yearmonth'] = df_all['yearmonth'].astype(int)
    
    # Check if sensor column exists (needed for harmonization)
    if 'sensor' not in df_all.columns:
        logger.info("ℹ Sensor column not found - skipping harmonization (using combined sensor medians)")
        logger.info(f"✓ Data ready: {len(csv_files)} files with {len(df_all)} observations")
        return
    
    if L7_EXCLUDE_POST_SLC:
        mask_l7_slc = (df_all['sensor'] == 'LANDSAT_7') & (df_all['yearmonth'] >= L7_SLC_FAILURE_DATE)
        logger.info(f"Excluding {mask_l7_slc.sum()} L7 post-SLC observations")
        df_all = df_all[~mask_l7_slc].copy()
    
    # L5→L7 calibration
    overlap_start_ym = L5_L7_OVERLAP[0]*100 + L5_L7_OVERLAP[1]
    overlap_end_ym = L5_L7_OVERLAP[2]*100 + L5_L7_OVERLAP[3]
    
    df_overlap = df_all[(df_all['yearmonth'] >= overlap_start_ym) & (df_all['yearmonth'] <= overlap_end_ym)]
    df_l5_overlap = df_overlap[df_overlap['sensor'] == 'LANDSAT_5']
    df_l7_overlap = df_overlap[df_overlap['sensor'] == 'LANDSAT_7']
    
    logger.info(f"L5→L7 overlap: {len(df_l5_overlap)} L5, {len(df_l7_overlap)} L7 obs")
    
    l5_to_l7_params = {}
    l5_to_l7_quality = {}
    
    for uid in df_all['UID'].unique():
        merged = pd.merge(
            df_l5_overlap[df_l5_overlap['UID'] == uid][['yearmonth', 'NDVI']],
            df_l7_overlap[df_l7_overlap['UID'] == uid][['yearmonth', 'NDVI']],
            on='yearmonth', suffixes=('_L5', '_L7')
        )
        
        if len(merged) < MIN_HARMONIZATION_SAMPLES:
            continue
        
        try:
            X = merged['NDVI_L5'].values.reshape(-1,1)
            y = merged['NDVI_L7'].values
            model = HuberRegressor(epsilon=1.35).fit(X, y)
            y_pred = model.predict(X)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            
            l5_to_l7_params[uid] = (model.coef_[0], model.intercept_)
            l5_to_l7_quality[uid] = {'r2': r2, 'rmse': rmse, 'n': len(merged)}
        except:
            pass
    
    # Global fallback
    merged_global = pd.merge(
        df_l5_overlap[['UID', 'yearmonth', 'NDVI']],
        df_l7_overlap[['UID', 'yearmonth', 'NDVI']],
        on=['UID', 'yearmonth'], suffixes=('_L5', '_L7')
    )
    
    if len(merged_global) > 0:
        X_global = merged_global['NDVI_L5'].values.reshape(-1,1)
        y_global = merged_global['NDVI_L7'].values
        global_model = HuberRegressor(epsilon=1.35).fit(X_global, y_global)
        l5_to_l7_global = (global_model.coef_[0], global_model.intercept_)
        logger.info(f"Global L5→L7: slope={l5_to_l7_global[0]:.4f}, intercept={l5_to_l7_global[1]:.4f}")
    else:
        l5_to_l7_global = (1.0, 0.0)
    
    # L7→L8 (Roy et al. 2016)
    l7_to_l8_slope, l7_to_l8_intercept = 0.9723, 0.0138
    
    # Apply harmonization
    df_all['NDVI_harmonized'] = df_all['NDVI']
    
    for uid in df_all['UID'].unique():
        params = l5_to_l7_params.get(uid, l5_to_l7_global)
        a1, b1 = params
        
        mask_l5 = (df_all['UID'] == uid) & (df_all['sensor'] == 'LANDSAT_5')
        if mask_l5.any():
            ndvi_l7_scale = df_all.loc[mask_l5, 'NDVI'] * a1 + b1
            df_all.loc[mask_l5, 'NDVI_harmonized'] = ndvi_l7_scale * l7_to_l8_slope + l7_to_l8_intercept
        
        mask_l7 = (df_all['UID'] == uid) & (df_all['sensor'] == 'LANDSAT_7')
        if mask_l7.any():
            df_all.loc[mask_l7, 'NDVI_harmonized'] = df_all.loc[mask_l7, 'NDVI'] * l7_to_l8_slope + l7_to_l8_intercept
    
    # Save outputs
    if l5_to_l7_quality:
        pd.DataFrame(l5_to_l7_quality).T.to_csv(Path(local_out_dir)/"harmonization_L5_L7_quality.csv")
    
    df_all.groupby('sensor')['yearmonth'].agg(['min', 'max', 'count']).to_csv(
        Path(local_out_dir)/"sensor_summary.csv"
    )
    
    # QC reports - completeness based on months that exist in data
    all_months_in_data = set(df_all['yearmonth'].unique())
    
    completeness = df_all.groupby('UID', group_keys=False).apply(
        lambda x: len(set(x['yearmonth'])) / len(all_months_in_data),
        include_groups=False
    )
    completeness.to_csv(Path(local_out_dir)/"temporal_completeness.csv")
    
    df_sorted = df_all.sort_values(['UID','yearmonth'])
    df_sorted['ndvi_diff'] = df_sorted.groupby('UID')['NDVI_harmonized'].diff()
    anomalies = df_sorted[df_sorted['ndvi_diff'].abs() > 0.3]
    anomalies.to_csv(Path(local_out_dir)/"anomalous_jumps.csv", index=False)
    
    # Save harmonized data
    for year in sorted(df_all['yearmonth'].unique() // 100):
        df_year = df_all[df_all['yearmonth'] // 100 == year]
        df_year.to_csv(Path(local_out_dir)/f"NDVI_harmonized_{year}.csv", index=False)
    
    logger.info(f"✓ Harmonization complete: {len(l5_to_l7_quality)} polygons calibrated")
    logger.info(f"✓ QC: Completeness={completeness.mean():.1%}, Anomalies={len(anomalies)}")

def main(gee_asset_id: str, incremental: bool = True):
    """
    Main workflow using GEE Export tasks (no memory limits).
    
    Args:
        gee_asset_id: GEE asset path (e.g., 'projects/ee-litepc/assets/ANAEv3_gt1ha')
        incremental: True=append new months, False=full extraction
    """
    try:
       
        out_dir = Path(DEFAULT_OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # STEP 1: Check for downloaded files
        has_files = download_from_drive(out_dir)
        
        # STEP 2: Process downloaded files (if any exist)
        if has_files and process_downloaded_csvs(out_dir):
            logger.info("✓ Processed downloaded files")
        
        # STEP 3: Check what months still need extraction
        polygons = load_polygons(gee_asset_id, limit=TEST_POLYGON_LIMIT)
        aoi = polygons.geometry().bounds()
        
        # Determine date range
        if START_DATE:
            start_year, start_month = START_DATE
        else:
            start_year, start_month = LANDSAT_START_YEAR, 1
        
        if END_DATE:
            end_year, end_month = END_DATE
        else:
            end_year, end_month = get_landsat_latest_complete_month()
        
        logger.info(f"Date range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        if TEST_POLYGON_LIMIT:
            logger.info(f"Polygon limit: {TEST_POLYGON_LIMIT}")

        # STEP 4: Check if extraction needed
        if incremental:
            needs_extraction = build_incremental_collection(aoi, out_dir, start_year, start_month, end_year, end_month)
        else:
            needs_extraction = True  # Full extraction always runs

        if needs_extraction is None:
            # All months already extracted
            logger.info("✓ All months extracted")
            harmonize_landsat_sensors(out_dir)
            logger.info("\n" + "="*60)
            logger.info("✓ PROCESSING COMPLETE")
            logger.info(f"Output files: {out_dir}")
            logger.info("  NDVI_Landsat_YYYY.csv - Monthly NDVI by polygon")
            logger.info("="*60)
            return
        
        # Export monthly tasks for missing months
        tasks = export_monthly_tasks(start_year, start_month, end_year, end_month, aoi, polygons)
        
        if not tasks:
            logger.warning("⚠ No tasks created - check date range and sensor availability")
            return
            
        logger.info(f"✓ Started {len(tasks)} monthly export tasks")
        
        logger.info("\n" + "="*60)
        logger.info("WORKFLOW:")
        logger.info("1. Tasks are running on GEE servers")
        logger.info("   Monitor: https://code.earthengine.google.com/tasks")
        logger.info("2. When tasks complete, download CSVs from Google Drive:")
        logger.info("   - Go to: https://drive.google.com")
        logger.info("   - Open folder: GEE_Landsat_NDVI")
        logger.info(f"   - Download all CSV files to: {out_dir.absolute()}")
        logger.info("3. Re-run this script to process and harmonize data")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Asset configuration
    GEE_ASSET_ID = "projects/ee-litepc/assets/ANAEv3_gt1ha"
    INCREMENTAL = True
    
    # Validate configuration
    if USE_CLOUD_STORAGE and GCS_BUCKET == 'your-gcs-bucket-name':
        logger.error("✗ USE_CLOUD_STORAGE=True requires valid GCS_BUCKET")
        exit(1)
    
    main(GEE_ASSET_ID, INCREMENTAL)
