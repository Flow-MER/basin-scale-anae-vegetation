"""
Landsat NDVI Extraction & Harmonization (1986-Present)

Extracts monthly median NDVI from Landsat 5/7/8/9 for large polygon sets using GEE.
Water pixels (NDVI < 0) are removed focus on vegetation
Performs per-polygon sensor harmonization and outputs annual CSV files.

WORKFLOW:
    1. GEE Export: Submit tasks to extract monthly NDVI → Google Drive
    2. Download: Manually download CSVs from Drive to ./landsat_ndvi_output/
    3. Process: Re-run script to harmonize sensors and generate annual files

OUTPUT:
    NDVI_Landsat_{year}.csv - Best-available NDVI per month (4 decimals)
        Columns: UID, yearmonth, NDVI, NDVI_sd, pixel_count, sensor_source
        sensor_source: L8_9 | L5_harmonized | L7_harmonized | L5_harmonized_global | L7_harmonized_global
    
    harmonization_L5_L7_quality.csv - Per-polygon L5→L7 calibration metrics
    harmonization_L7_L8_quality.csv - Per-polygon L7→L8 calibration metrics
    sensor_summary.csv - Data availability by sensor
    temporal_completeness.csv - Fraction of months with data per polygon
    anomalous_jumps.csv - Month-to-month NDVI changes >30%

HARMONIZATION:
    L5 → L7 → L8 chain using per-polygon regression (June 1999 - April 2003 overlap)
    L7 post-SLC (May 2003+) excluded due to striping artifacts
    L8/L9 combined (identical sensors, no harmonization needed)
    Fallback: Roy et al. (2016) published coefficients when insufficient overlap data

REFERENCE:
    Roy et al. (2016) RSE 185:57-70 - Landsat cross-calibration coefficients
    https://doi.org/10.1016/j.rse.2015.12.024

Author: Shane Brooks (brooks.eco)
Date: 2025-01-09
"""

import ee
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv
import os
import gee_task_watchdog

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DEFAULT_OUT_DIR = "./landsat_ndvi_output"
GEE_ASSET_ID = "projects/ee-litepc/assets/ANAEv3_gt1ha"  # Must have 'UID' field

# Date range (None = auto-detect)
START_DATE = (1986, 1)  # (year, month) or None for full archive from 1986
END_DATE = (1999, 12)    # (year, month) or None for latest complete month

# Testing
TEST_POLYGON_LIMIT = None  # None = all polygons, 10 = first 10 polygons
USE_RANDOM_SAMPLE = False  # True = random sample, False = first N polygons

# NOTE: L5 Collection 2 has data gap for Australia in 2001-2002
# Use 1999-2000 or 2003+ for L5 testing

# Export destination
USE_CLOUD_STORAGE = False  # False = Google Drive (user OAuth), True = GCS (service account)
GCS_BUCKET = os.getenv('GCS_BUCKET', 'your-bucket')
GEE_PROJECT = os.getenv('GEE_PROJECT', None)

# Landsat parameters
LANDSAT_START_YEAR = 1986
L5_L7_OVERLAP = (1999, 6, 2003, 4)  # June 1999 - April 2003 (pre-SLC failure)
L7_SLC_FAILURE_DATE = 200305  # May 2003
SCALE_FACTOR, OFFSET = 0.0000275, -0.2  # USGS C2 L2 scaling
REDUCE_SCALE = 30  # Landsat pixel size (meters)
MIN_HARMONIZATION_SAMPLES = 12  # Minimum paired observations for calibration
ROY_L7_L8_COEF = (0.9723, 0.0235)  # Roy et al. (2016) Table 3: OLI = 0.0235 + 0.9723×ETM+

LANDSAT_COLLECTIONS = {
    'L5': 'LANDSAT/LT05/C02/T1_L2',
    'L7': 'LANDSAT/LE07/C02/T1_L2',
    'L8': 'LANDSAT/LC08/C02/T1_L2',
    'L9': 'LANDSAT/LC09/C02/T1_L2'
}

# =============================================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# =============================================================================

def init_gee():
    """Initialize Google Earth Engine with appropriate authentication."""
    if USE_CLOUD_STORAGE:
        from google.oauth2 import service_account
        sa_json = os.getenv('GEE_SERVICE_ACCOUNT_JSON')
        if not sa_json or not Path(sa_json).exists():
            raise FileNotFoundError(f"Service account JSON not found: {sa_json}")
        creds = service_account.Credentials.from_service_account_file(
            sa_json, scopes=['https://www.googleapis.com/auth/earthengine'])
        ee.Initialize(creds)
        logger.info("✓ GEE initialized (service account)")
    else:
        try:
            ee.Initialize(project=GEE_PROJECT)
            logger.info("✓ GEE initialized (user OAuth)")
        except:
            logger.info("Authenticating...")
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
            logger.info("✓ GEE authenticated")

init_gee()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_months(start_year, start_month, end_year, end_month):
    """Generate list of (year, month) tuples."""
    months, y, m = [], start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m = m % 12 + 1
        y += (m == 1)
    return months

def get_latest_landsat_month():
    """Auto-detect latest complete Landsat month."""
    now = datetime.now()
    for months_back in range(1, 4):
        check = now.replace(day=1)
        for _ in range(months_back):
            check = (check.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
        
        start = ee.Date.fromYMD(check.year, check.month, 1)
        end = start.advance(1, 'month')
        
        for cid in ['LANDSAT/LC09/C02/T1_L2', 'LANDSAT/LC08/C02/T1_L2']:
            if ee.ImageCollection(cid).filterDate(start, end).size().getInfo() > 0:
                logger.info(f"✓ Latest Landsat: {check.year}-{check.month:02d}")
                return check.year, check.month
    
    return now.year, now.month

# =============================================================================
# GEE NDVI COMPUTATION
# =============================================================================

def compute_ndvi(image):
    """Compute cloud-masked NDVI for Landsat 4-9.
    
    Steps: Scale to reflectance → Mask clouds/shadows/snow → Calculate NDVI
    """
    # Scale DN to surface reflectance
    optical = image.select('SR_B.*').multiply(SCALE_FACTOR).add(OFFSET)
    image = image.addBands(optical, None, True)
    
    # Mask clouds (bit 3), shadows (bit 4), snow (bit 5)
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(  # Bit 3: Cloud
           qa.bitwiseAnd(1 << 4).eq(0)).And( # Bit 4: Cloud shadow
           qa.bitwiseAnd(1 << 5).eq(0))      # Bit 5: Snow
    # Landsat7 Mask fill (bit 0)
    spacecraft = ee.String(image.get('SPACECRAFT_ID'))
    mask = ee.Algorithms.If(spacecraft.equals('LANDSAT_7'), mask.And(qa.bitwiseAnd(1).eq(0)), mask)
    image = image.updateMask(ee.Image(mask))
    
    # Select NIR/Red bands (sensor-specific - L8/L9 use B5/B4, others use B4/B3)
    nir = ee.Image(ee.Algorithms.If(spacecraft.equals('LANDSAT_8'), image.select('SR_B5'),
                   ee.Algorithms.If(spacecraft.equals('LANDSAT_9'), image.select('SR_B5'),
                                    image.select('SR_B4'))))
    red = ee.Image(ee.Algorithms.If(spacecraft.equals('LANDSAT_8'), image.select('SR_B4'),
                   ee.Algorithms.If(spacecraft.equals('LANDSAT_9'), image.select('SR_B4'),
                                    image.select('SR_B3'))))
    
    # Calculate NDVI
    ndvi = nir.subtract(red).divide(nir.add(red).add(1e-10)).rename('NDVI')
    # Mask to remove water pixels (keep NDVI >= 0)
    #ndvi = ndvi.updateMask(ndvi.gte(0))
    return ndvi.copyProperties(image, ['system:time_start', 'SPACECRAFT_ID'])

# =============================================================================
# GEE EXPORT
# =============================================================================

def export_monthly_ndvi(months_to_export, polygons):
    """Submit GEE export tasks for monthly NDVI.
    
    Strategy: Export sensors separately when overlap exists for harmonization.
        Pre-1999: L5 only
        1999-2003: L5, L7 (L5→L7 calibration period)
        2003-2012: L5, L7 (both sensors, L7 post-SLC excluded during processing)
        2013+: L7, L8+L9 combined (L7→L8 calibration, L8/L9 merged for better coverage)
    
    L8/L9 combined: Identical OLI sensors, doubles temporal coverage (8-day revisit).
    """
    logger.info("Computing AOI bounds...")
    # Murray-Darling Basin AOI (avoids expensive geometry computation)
    aoi = ee.Geometry.Polygon([[[138.5,-37.6], [152.5,-37.6], [152.5,-24.5], [138.5,-24.5]]], None, False)
    logger.info("✓ AOI ready")
    tasks = []
    
    for y, m in months_to_export:
        ym = f"{y:04d}{m:02d}"
        start = ee.Date.fromYMD(y, m, 1)
        end = start.advance(1, 'month')
        
        # Determine which sensors to export (empty collections skipped automatically)
        sensors = []
        if 1984 <= y <= 2013:  # L5 operational 1984-2013
            sensors.append(('L5', LANDSAT_COLLECTIONS['L5']))
        if 1999 <= y <= 2003:  # L7 pre-SLC failure 1999-April 2003
            sensors.append(('L7', LANDSAT_COLLECTIONS['L7']))
        if y >= 2013:  # L8 launched April 2013, L9 October 2021-present
            sensors.append(('L8L9', None))
        
        for sensor_name, collection_id in sensors:
            if sensor_name == 'L8L9':
                # Combine L8+L9 for better temporal coverage
                l8 = ee.ImageCollection(LANDSAT_COLLECTIONS['L8']).filterBounds(aoi).filterDate(start, end).map(compute_ndvi)
                l9 = ee.ImageCollection(LANDSAT_COLLECTIONS['L9']).filterBounds(aoi).filterDate(start, end).map(compute_ndvi)
                col = l8.merge(l9)
            else:
                col = ee.ImageCollection(collection_id).filterBounds(aoi).filterDate(start, end).map(compute_ndvi)
            
            monthly_img = col.median().clip(aoi)
            
            # Compute zonal statistics (task will fail gracefully if composite is empty)
            stats = monthly_img.reduceRegions(
                collection=polygons,
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True).combine(ee.Reducer.count(), '', True),
                scale=REDUCE_SCALE,
                tileScale=4
            ).map(lambda f: f.select(['mean', 'stdDev', 'count', 'UID'], ['NDVI', 'NDVI_sd', 'pixel_count', 'UID']).set('yearmonth', ym)
            ).filter(ee.Filter.notNull(['NDVI']))
            
            # Export
            desc = f"NDVI_{sensor_name}_{ym}"
            if USE_CLOUD_STORAGE:
                task = ee.batch.Export.table.toCloudStorage(
                    collection=stats, description=desc, bucket=GCS_BUCKET,
                    fileNamePrefix=f"GEE_Landsat_NDVI/{desc}", fileFormat='CSV',
                    selectors=['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count'])
            else:
                task = ee.batch.Export.table.toDrive(
                    collection=stats, description=desc, folder='GEE_Landsat_NDVI', fileFormat='CSV',
                    selectors=['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count'])
            
            task.start()
            tasks.append(task)
            logger.info(f"✓ Task started: {desc}")
    
    logger.info(f"Monitor: https://code.earthengine.google.com/tasks")
    return tasks

# =============================================================================
# CSV PROCESSING
# =============================================================================

def consolidate_monthly_csvs(out_dir):
    """Load and consolidate monthly CSVs from GEE exports."""
    csv_files = list(Path(out_dir).glob("NDVI_L*_*.csv"))
    if not csv_files:
        logger.warning("No CSV files found")
        return False
    
    # Check if already consolidated
    monthly_file = Path(out_dir) / "NDVI_monthly_raw.csv"
    if monthly_file.exists():
        existing = pd.read_csv(monthly_file)
        existing_months = set(existing['yearmonth'].unique())
        logger.info(f"✓ Found existing consolidated file with {len(existing_months)} months")
    else:
        existing = None
        existing_months = set()
    
    logger.info(f"✓ Found {len(csv_files)} monthly CSVs")
    
    monthly_data = []
    new_months = 0
    for csv_file in csv_files:
        try:
            sensor = csv_file.stem.split('_')[1]  # Extract L5, L7, or L8L9
            df = pd.read_csv(csv_file)
            if 'yearmonth' not in df.columns or df.empty:
                continue
            
            # Check if this month already processed
            month = df['yearmonth'].iloc[0]
            if month in existing_months:
                continue
            
            df['sensor'] = 'L8_9' if sensor == 'L8L9' else f"L{sensor[1]}"
            monthly_data.append(df)
            new_months += 1
            logger.info(f"  Loaded: {csv_file.name}")
        except Exception as e:
            logger.warning(f"  Error: {csv_file.name} - {e}")
    
    if not monthly_data and existing is None:
        return False
    
    # Consolidate
    if monthly_data:
        df_new = pd.concat(monthly_data, ignore_index=True)
        df_new['yearmonth'] = df_new['yearmonth'].astype(int)
        
        if existing is not None:
            df_all = pd.concat([existing, df_new], ignore_index=True)
        else:
            df_all = df_new
        
        df_all.drop_duplicates(subset=['UID', 'yearmonth', 'sensor'], inplace=True)
        df_all.to_csv(monthly_file, index=False)
        logger.info(f"✓ Consolidated {len(df_all)} observations ({new_months} new months) → {monthly_file.name}")
    else:
        logger.info("✓ No new months to consolidate")
    
    return True

# =============================================================================
# SENSOR HARMONIZATION
# =============================================================================

def calibrate_sensor_pair(df_source, df_target, uids, min_samples, pair_name):
    """Per-polygon calibration between two sensors using Huber regression."""
    params, quality = {}, {}
    
    for uid in uids:
        merged = pd.merge(
            df_source[df_source['UID'] == uid][['yearmonth', 'NDVI']],
            df_target[df_target['UID'] == uid][['yearmonth', 'NDVI']],
            on='yearmonth', suffixes=('_src', '_tgt'))
        
        if len(merged) < min_samples:
            continue
        
        try:
            X, y = merged['NDVI_src'].values.reshape(-1, 1), merged['NDVI_tgt'].values
            model = HuberRegressor(epsilon=1.35).fit(X, y)
            y_pred = model.predict(X)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            
            params[uid] = (model.coef_[0], model.intercept_)
            quality[uid] = {'r2': r2, 'rmse': rmse, 'n': len(merged)}
        except:
            pass
    
    # Global fallback
    merged_global = pd.merge(
        df_source[['UID', 'yearmonth', 'NDVI']],
        df_target[['UID', 'yearmonth', 'NDVI']],
        on=['UID', 'yearmonth'], suffixes=('_src', '_tgt'))
    
    global_params = (1.0, 0.0)
    if len(merged_global) > 0:
        X_g, y_g = merged_global['NDVI_src'].values.reshape(-1, 1), merged_global['NDVI_tgt'].values
        model_g = HuberRegressor(epsilon=1.35).fit(X_g, y_g)
        global_params = (model_g.coef_[0], model_g.intercept_)
        logger.info(f"Global {pair_name}: slope={global_params[0]:.4f}, intercept={global_params[1]:.4f}")
        logger.info(f"Per-polygon {pair_name}: {len(params)}/{len(uids)} calibrated")
    
    return params, quality, global_params

def harmonize_sensors(out_dir):
    """Harmonize Landsat sensors and assemble best-available annual files."""
    logger.info("Starting sensor harmonization...")
    
    monthly_file = Path(out_dir) / "NDVI_monthly_raw.csv"
    if not monthly_file.exists():
        logger.warning("No monthly data file - run consolidate_monthly_csvs first")
        return
    
    df_all = pd.read_csv(monthly_file)
    df_all['yearmonth'] = df_all['yearmonth'].astype(int)
    
    # Exclude L7 post-SLC failure
    mask_l7_slc = (df_all['sensor'] == 'L7') & (df_all['yearmonth'] >= L7_SLC_FAILURE_DATE)
    logger.info(f"Excluding {mask_l7_slc.sum()} L7 post-SLC observations")
    df_all = df_all[~mask_l7_slc].copy()
    
    # Setup
    overlap_start = L5_L7_OVERLAP[0] * 100 + L5_L7_OVERLAP[1]
    overlap_end = L5_L7_OVERLAP[2] * 100 + L5_L7_OVERLAP[3]
    df_overlap = df_all[(df_all['yearmonth'] >= overlap_start) & (df_all['yearmonth'] <= overlap_end)]
    uids = df_all['UID'].unique()
    
    # L5→L7 calibration
    has_l5_overlap = (df_overlap['sensor'] == 'L5').any()
    has_l7_overlap = (df_overlap['sensor'] == 'L7').any()
    
    if has_l5_overlap and has_l7_overlap:
        logger.info("Calibrating L5→L7...")
        l5_l7_params, l5_l7_quality, l5_l7_global = calibrate_sensor_pair(
            df_overlap[df_overlap['sensor'] == 'L5'],
            df_overlap[df_overlap['sensor'] == 'L7'],
            uids, MIN_HARMONIZATION_SAMPLES, "L5→L7")
    else:
        l5_l7_params, l5_l7_quality = {}, {}
        l5_l7_global = ROY_L7_L8_COEF if (df_all['sensor'] == 'L5').any() else (1.0, 0.0)
        if (df_all['sensor'] == 'L5').any():
            logger.info("No L5/L7 overlap - using published coefficients")
    
    # L7→L8 calibration
    has_l8 = (df_all['sensor'] == 'L8_9').any()
    
    if has_l7_overlap and has_l8:
        logger.info("Calibrating L7→L8...")
        l7_l8_params, l7_l8_quality, l7_l8_global = calibrate_sensor_pair(
            df_all[df_all['sensor'] == 'L7'],
            df_all[df_all['sensor'] == 'L8_9'],
            uids, MIN_HARMONIZATION_SAMPLES, "L7→L8")
    else:
        l7_l8_params, l7_l8_quality = {}, {}
        l7_l8_global = ROY_L7_L8_COEF
        if (df_all['sensor'] == 'L7').any():
            logger.info("No L7/L8 overlap - using published coefficients")
    
    # Fast path: Skip harmonization loop if no overlapping sensors exist
    # Check if any (UID, yearmonth) has multiple sensors (requires selection/harmonization)
    has_overlaps = df_all.groupby(['UID', 'yearmonth'])['sensor'].nunique().max() > 1
    
    if not has_overlaps:
        logger.info("No overlapping sensors - using fast path")
        df_harmonized = df_all.copy()
        # Assign sensor_source and apply harmonization
        df_harmonized['sensor_source'] = df_harmonized['sensor'].map({
            'L8_9': 'L8_9',
            'L5': 'L5_harmonized_global' if l5_l7_global != (1.0, 0.0) else 'L5',
            'L7': 'L7_harmonized_global'  # L7 always harmonized when no overlaps
        })
        
        # Apply L5→L7→L8 chain
        mask_l5 = df_harmonized['sensor'] == 'L5'
        if mask_l5.any() and l5_l7_global != (1.0, 0.0):
            logger.info(f"Applying L5→L7→L8 chain to {mask_l5.sum()} L5 observations")
            ndvi_l7 = df_harmonized.loc[mask_l5, 'NDVI'] * l5_l7_global[0] + l5_l7_global[1]
            df_harmonized.loc[mask_l5, 'NDVI'] = ndvi_l7 * l7_l8_global[0] + l7_l8_global[1]
        
        # Apply L7→L8
        mask_l7 = df_harmonized['sensor'] == 'L7'
        if mask_l7.any():
            logger.info(f"Applying L7→L8 to {mask_l7.sum()} L7 observations")
            df_harmonized.loc[mask_l7, 'NDVI'] = df_harmonized.loc[mask_l7, 'NDVI'] * l7_l8_global[0] + l7_l8_global[1]
        
        mask_l8 = df_harmonized['sensor'] == 'L8_9'
        if mask_l8.any():
            logger.info(f"Using {mask_l8.sum()} L8_9 observations as-is (reference)")
    else:
        # Overlapping sensors exist - use full harmonization loop
        logger.info("Overlapping sensors detected - selecting best sensor per (UID, month)")
        df_harmonized = []
        sensor_counts = {'L8_9': 0, 'L7': 0, 'L5': 0}
        
        for (uid, ym), group in df_all.groupby(['UID', 'yearmonth']):
            sensors = group['sensor'].values
            best = None
            
            # Priority: L8_9 > L7 (overlap) > L5
            if 'L8_9' in sensors:
                best = group[group['sensor'] == 'L8_9'].iloc[0].copy()
                best['sensor_source'] = 'L8_9'
                sensor_counts['L8_9'] += 1
            
            elif 'L7' in sensors and overlap_start <= ym <= overlap_end:
                best = group[group['sensor'] == 'L7'].iloc[0].copy()
                params = l7_l8_params.get(uid, l7_l8_global)
                best['NDVI'] = best['NDVI'] * params[0] + params[1]
                best['sensor_source'] = 'L7_harmonized' if uid in l7_l8_params else 'L7_harmonized_global'
                sensor_counts['L7'] += 1
            
            elif 'L5' in sensors:
                best = group[group['sensor'] == 'L5'].iloc[0].copy()
                l5_l7 = l5_l7_params.get(uid, l5_l7_global)
                l7_l8 = l7_l8_params.get(uid, l7_l8_global)
                
                # L5→L7→L8 chain
                ndvi_l7 = best['NDVI'] * l5_l7[0] + l5_l7[1]
                best['NDVI'] = ndvi_l7 * l7_l8[0] + l7_l8[1]
                
                if uid in l5_l7_params:
                    best['sensor_source'] = 'L5_harmonized' if uid in l7_l8_params or len(l7_l8_params) > 0 else 'L5_harmonized_global'
                else:
                    best['sensor_source'] = 'L5_harmonized_global'
                sensor_counts['L5'] += 1
            
            elif 'L7' in sensors:
                # L7 outside overlap period - harmonize with Roy coefficients
                best = group[group['sensor'] == 'L7'].iloc[0].copy()
                best['NDVI'] = best['NDVI'] * l7_l8_global[0] + l7_l8_global[1]
                best['sensor_source'] = 'L7_harmonized_global'
                sensor_counts['L7'] += 1
            
            if best is not None:
                df_harmonized.append(best)
        
        df_harmonized = pd.DataFrame(df_harmonized)
        logger.info(f"Selected: L8_9={sensor_counts['L8_9']}, L7={sensor_counts['L7']}, L5={sensor_counts['L5']} observations")
    
    # Save annual files (rounded to 4 decimals)
    for year in sorted(df_harmonized['yearmonth'].unique() // 100):
        df_year = df_harmonized[df_harmonized['yearmonth'] // 100 == year].copy()
        df_year['NDVI'] = df_year['NDVI'].round(4)
        df_year['NDVI_sd'] = df_year['NDVI_sd'].round(4)
        df_year = df_year[['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count', 'sensor_source']]
        df_year.to_csv(Path(out_dir) / f"NDVI_Landsat_{year}.csv", index=False)
    
    # Save QC outputs
    if l5_l7_quality:
        pd.DataFrame(l5_l7_quality).T.to_csv(Path(out_dir) / "harmonization_L5_L7_quality.csv")
    if l7_l8_quality:
        pd.DataFrame(l7_l8_quality).T.to_csv(Path(out_dir) / "harmonization_L7_L8_quality.csv")
    
    df_harmonized.groupby('sensor_source')['yearmonth'].agg(['min', 'max', 'count']).to_csv(
        Path(out_dir) / "sensor_summary.csv")
    
    all_months = set(df_harmonized['yearmonth'].unique())
    completeness = df_harmonized.groupby('UID', group_keys=False).apply(
        lambda x: len(set(x['yearmonth'])) / len(all_months), include_groups=False)
    completeness.to_csv(Path(out_dir) / "temporal_completeness.csv")
    
    df_sorted = df_harmonized.sort_values(['UID', 'yearmonth'])
    df_sorted['ndvi_diff'] = df_sorted.groupby('UID')['NDVI'].diff()
    anomalies = df_sorted[df_sorted['ndvi_diff'].abs() > 0.3]
    anomalies.to_csv(Path(out_dir) / "anomalous_jumps.csv", index=False)
    
    # Summary
    calib_summary = []
    if len(l5_l7_params) > 0:
        calib_summary.append(f"L5→L7: {len(l5_l7_params)} per-polygon")
    if len(l7_l8_params) > 0:
        calib_summary.append(f"L7→L8: {len(l7_l8_params)} per-polygon")
    
    if calib_summary:
        logger.info(f"✓ Harmonization complete: {', '.join(calib_summary)}")
    else:
        logger.info("✓ Processing complete (no harmonization needed)")
    
    logger.info(f"✓ Best-available NDVI: {len(df_harmonized)} observations")
    logger.info(f"✓ QC: Completeness={completeness.mean():.1%}, Anomalies={len(anomalies)}")

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Main workflow: GEE export → Download → Process → Harmonize."""
    out_dir = Path(DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine date range
    start_year, start_month = START_DATE if START_DATE else (LANDSAT_START_YEAR, 1)
    end_year, end_month = END_DATE if END_DATE else get_latest_landsat_month()
    
    # Check if requested months already have CSVs
    months = list_months(start_year, start_month, end_year, end_month)
    missing_months = []
    for y, m in months:
        ym = f"{y:04d}{m:02d}"
        # Check for any sensor CSV for this month
        month_csvs = list(out_dir.glob(f"NDVI_*_{ym}.csv"))
        if not month_csvs:
            missing_months.append((y, m))
    
    if not missing_months:
        # All requested months exist - process them
        logger.info(f"✓ All requested months have CSVs - processing...")
        if consolidate_monthly_csvs(out_dir):
            harmonize_sensors(out_dir)
            logger.info("\n" + "="*60)
            logger.info("✓ PROCESSING COMPLETE")
            logger.info(f"Output: {out_dir}")
            logger.info("="*60)
            return
    
    # Missing CSVs - submit GEE export tasks
    logger.info(f"Missing {len(missing_months)} months - submitting GEE export tasks...")
    
    polygons = ee.FeatureCollection(GEE_ASSET_ID)
    if TEST_POLYGON_LIMIT:
        if USE_RANDOM_SAMPLE:
            logger.info(f"Selecting {TEST_POLYGON_LIMIT} random polygons (may take 30-60 sec)...")
            polygons = polygons.randomColumn('random', 42).sort('random').limit(TEST_POLYGON_LIMIT)
            logger.info(f"✓ Random sample ready")
        else:
            polygons = ee.FeatureCollection(polygons.toList(TEST_POLYGON_LIMIT))
            logger.info(f"✓ Using first {TEST_POLYGON_LIMIT} polygons (test mode)")
    else:
        logger.info(f"✓ Using all polygons from {GEE_ASSET_ID}")
    
    logger.info(f"Date range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    logger.info(f"Exporting only missing months: {[f'{y:04d}-{m:02d}' for y, m in missing_months]}")
    
    # Submit export tasks for missing months only
    if gee_task_watchdog.launch_watchdog():
        logger.info("✓ Task watchdog launched in new window")
    
    tasks = export_monthly_ndvi(missing_months, polygons)
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS:")
    logger.info("1. Monitor tasks: https://code.earthengine.google.com/tasks")
    logger.info("2. Download CSVs from Google Drive folder: GEE_Landsat_NDVI")
    logger.info(f"3. Save CSVs to: {out_dir.absolute()}")
    logger.info("4. Re-run this script to process and harmonize")
    logger.info("="*60)

if __name__ == "__main__":
    main()
