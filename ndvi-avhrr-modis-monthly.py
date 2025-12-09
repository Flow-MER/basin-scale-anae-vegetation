"""
AVHRR/MODIS NDVI Extraction & Harmonization (1986-Present)

Extracts monthly median NDVI from AVHRR (1986-2013) and MODIS (2000-present).
Performs per-polygon sensor harmonization and outputs annual CSV files.

WORKFLOW:
    1. GEE Export: Submit tasks to extract monthly NDVI → Google Drive
    2. Download: Manually download CSVs from Drive to ./avhrr_modis_output/
    3. Process: Re-run script to harmonize sensors and generate annual files

OUTPUT:
    NDVI_Harmonized_{year}.csv - Best-available NDVI per month (4 decimals)
        Columns: UID, yearmonth, NDVI, NDVI_sd, pixel_count, sensor_source
        sensor_source: MODIS | AVHRR_harmonized | AVHRR_harmonized_global
    
    harmonization_quality.csv - Per-polygon AVHRR→MODIS calibration metrics
    sensor_summary.csv - Data availability by sensor
    temporal_completeness.csv - Fraction of months with data per polygon
    anomalous_jumps.csv - Month-to-month NDVI changes >30%

HARMONIZATION:
    AVHRR → MODIS using per-polygon regression (2000-2013 overlap)
    Fallback: Global calibration when insufficient overlap data

Author: Shane Brooks (brooks.eco)
Date: 2025-01-10
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

DEFAULT_OUT_DIR = "./avhrr_modis_output"
GEE_ASSET_ID = "projects/ee-litepc/assets/ANAEv3_gt1ha"

# Date range
START_DATE = (1986, 1)  # AVHRR start
END_DATE = (2025, 12)  # None = auto-detect latest MODIS

# Testing
TEST_POLYGON_LIMIT = None
USE_RANDOM_SAMPLE = False

# Export
USE_CLOUD_STORAGE = False
GCS_BUCKET = os.getenv('GCS_BUCKET', 'your-bucket')
GEE_PROJECT = os.getenv('GEE_PROJECT', None)

# Sensor parameters
AVHRR_END = (2013, 12)
MODIS_START = (2000, 3)
OVERLAP_PERIOD = (2000, 3, 2013, 12)
SCALE_FACTOR = 0.0001
REDUCE_SCALE = 250  # MODIS resolution
MIN_HARMONIZATION_SAMPLES = 24  # Need 24 months for calibration

SENSOR_COLLECTIONS = {
    'AVHRR': 'NOAA/CDR/AVHRR/NDVI/V5',
    'MODIS': 'MODIS/061/MOD13Q1'
}

# =============================================================================
# GEE INITIALIZATION
# =============================================================================

def init_gee():
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

init_gee()

# =============================================================================
# UTILITIES
# =============================================================================

def list_months(start_year, start_month, end_year, end_month):
    months, y, m = [], start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m = m % 12 + 1
        y += (m == 1)
    return months

def get_latest_modis_month():
    now = datetime.now()
    for months_back in range(1, 4):
        check = now.replace(day=1)
        for _ in range(months_back):
            check = (check.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
        start = ee.Date.fromYMD(check.year, check.month, 1)
        end = start.advance(1, 'month')
        if ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start, end).size().getInfo() > 0:
            logger.info(f"✓ Latest MODIS: {check.year}-{check.month:02d}")
            return check.year, check.month
    return now.year, now.month

# =============================================================================
# GEE EXPORT
# =============================================================================

def export_monthly_ndvi(months_to_export, polygons):
    """Submit GEE export tasks for monthly NDVI."""
    # Murray-Darling Basin AOI
    aoi = ee.Geometry.Polygon([[[138.5,-37.6], [152.5,-37.6], [152.5,-24.5], [138.5,-24.5]]], None, False)
    tasks = []
    
    for y, m in months_to_export:
        ym = f"{y:04d}{m:02d}"
        start = ee.Date.fromYMD(y, m, 1)
        end = start.advance(1, 'month')
        
        # Determine which sensors to export
        sensors_to_export = []
        
        # AVHRR: 1986-2013
        if y <= AVHRR_END[0] and (y < AVHRR_END[0] or m <= AVHRR_END[1]):
            sensors_to_export.append('AVHRR')
        
        # MODIS: 2000-present
        if y >= MODIS_START[0] and (y > MODIS_START[0] or m >= MODIS_START[1]):
            sensors_to_export.append('MODIS')
        
        # Export each sensor
        for sensor_name in sensors_to_export:
            if sensor_name == 'AVHRR':
                coll = ee.ImageCollection(SENSOR_COLLECTIONS['AVHRR']).filterBounds(aoi).filterDate(start, end)
                # Bitmask filter: exclude clouds (bit 1) and water (bit 3)
                def mask_avhrr(img):
                    qa = img.select('QA')
                    cloud_mask = qa.bitwiseAnd(1 << 1).eq(0)  # bit 1 = 0 (not cloudy)
                    water_mask = qa.bitwiseAnd(1 << 3).eq(0)  # bit 3 = 0 (not water)
                    return img.updateMask(cloud_mask.And(water_mask))
                coll = coll.map(mask_avhrr)
                monthly_img = coll.select('NDVI').mean().multiply(SCALE_FACTOR).clip(aoi)
            else:  # MODIS
                coll = ee.ImageCollection(SENSOR_COLLECTIONS['MODIS']).filterBounds(aoi).filterDate(start, end)
                # Bitmask filter: SummaryQA bits 0-1 <= 1 (good/marginal, exclude snow/cloud)
                def mask_modis(img):
                    qa = img.select('SummaryQA')
                    return img.updateMask(qa.bitwiseAnd(3).lte(1))  # bits 0-1: 0 or 1
                coll = coll.map(mask_modis)
                monthly_img = coll.select('NDVI').mean().multiply(SCALE_FACTOR).clip(aoi)
            
            # Vegetation mask: NDVI >= 0 (exclude water)
            monthly_img = monthly_img.updateMask(monthly_img.gte(0))
            
            # Zonal statistics
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
# CSV PROCESSING
# =============================================================================

def consolidate_monthly_csvs(out_dir):
    csv_files = list(Path(out_dir).glob("NDVI_*_*.csv"))
    if not csv_files:
        logger.warning("No CSV files found")
        return False
    
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
            sensor = csv_file.stem.split('_')[1]
            df = pd.read_csv(csv_file)
            if 'yearmonth' not in df.columns or df.empty:
                continue
            
            month = df['yearmonth'].iloc[0]
            if month in existing_months:
                continue
            
            df['sensor'] = sensor
            monthly_data.append(df)
            new_months += 1
            logger.info(f"  Loaded: {csv_file.name}")
        except Exception as e:
            logger.warning(f"  Error: {csv_file.name} - {e}")
    
    if not monthly_data and existing is None:
        return False
    
    if monthly_data:
        df_new = pd.concat(monthly_data, ignore_index=True)
        df_new['yearmonth'] = df_new['yearmonth'].astype(int)
        
        if existing is not None:
            df_all = pd.concat([existing, df_new], ignore_index=True)
        else:
            df_all = df_new
        
        df_all.drop_duplicates(subset=['UID', 'yearmonth', 'sensor'], inplace=True)
        df_all.to_csv(monthly_file, index=False)
        logger.info(f"✓ Consolidated {len(df_all)} observations ({new_months} new months)")
    else:
        logger.info("✓ No new months to consolidate")
    
    return True

# =============================================================================
# HARMONIZATION
# =============================================================================

def harmonize_sensors(out_dir):
    logger.info("Starting AVHRR→MODIS harmonization...")
    
    monthly_file = Path(out_dir) / "NDVI_monthly_raw.csv"
    if not monthly_file.exists():
        logger.warning("No monthly data file")
        return
    
    df_all = pd.read_csv(monthly_file)
    df_all['yearmonth'] = df_all['yearmonth'].astype(int)
    
    # Setup overlap period
    overlap_start = OVERLAP_PERIOD[0] * 100 + OVERLAP_PERIOD[1]
    overlap_end = OVERLAP_PERIOD[2] * 100 + OVERLAP_PERIOD[3]
    df_overlap = df_all[(df_all['yearmonth'] >= overlap_start) & (df_all['yearmonth'] <= overlap_end)]
    uids = df_all['UID'].unique()
    
    # AVHRR→MODIS calibration
    has_avhrr = (df_overlap['sensor'] == 'AVHRR').any()
    has_modis = (df_overlap['sensor'] == 'MODIS').any()
    
    if has_avhrr and has_modis:
        logger.info("Calibrating AVHRR→MODIS...")
        params, quality, global_params = calibrate_sensor_pair(
            df_overlap[df_overlap['sensor'] == 'AVHRR'],
            df_overlap[df_overlap['sensor'] == 'MODIS'],
            uids)
    else:
        params, quality = {}, {}
        global_params = (1.0, 0.0)
        logger.info("No overlap data - using identity transform")
    
    # Apply harmonization
    df_harmonized = []
    sensor_counts = {'MODIS': 0, 'AVHRR': 0}
    
    for (uid, ym), group in df_all.groupby(['UID', 'yearmonth']):
        sensors = group['sensor'].values
        
        if 'MODIS' in sensors:
            best = group[group['sensor'] == 'MODIS'].iloc[0].copy()
            best['sensor_source'] = 'MODIS'
            sensor_counts['MODIS'] += 1
        elif 'AVHRR' in sensors:
            best = group[group['sensor'] == 'AVHRR'].iloc[0].copy()
            calib = params.get(uid, global_params)
            best['NDVI'] = best['NDVI'] * calib[0] + calib[1]
            best['sensor_source'] = 'AVHRR_harmonized' if uid in params else 'AVHRR_harmonized_global'
            sensor_counts['AVHRR'] += 1
        else:
            continue
        
        df_harmonized.append(best)
    
    df_harmonized = pd.DataFrame(df_harmonized)
    logger.info(f"Selected: MODIS={sensor_counts['MODIS']}, AVHRR={sensor_counts['AVHRR']}")
    
    # Save annual files (rounded to 4 decimals)
    for year in sorted(df_harmonized['yearmonth'].unique() // 100):
        df_year = df_harmonized[df_harmonized['yearmonth'] // 100 == year].copy()
        df_year['NDVI'] = df_year['NDVI'].round(4)
        df_year['NDVI_sd'] = df_year['NDVI_sd'].round(4)
        df_year = df_year[['UID', 'yearmonth', 'NDVI', 'NDVI_sd', 'pixel_count', 'sensor_source']]
        df_year.to_csv(Path(out_dir) / f"NDVI_Harmonized_{year}.csv", index=False)
    
    # Save QC outputs
    if quality:
        pd.DataFrame(quality).T.to_csv(Path(out_dir) / "harmonization_quality.csv")
    
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
    
    logger.info(f"✓ Harmonization complete: {len(params)} per-polygon calibrations")
    logger.info(f"✓ Best-available NDVI: {len(df_harmonized)} observations")
    logger.info(f"✓ QC: Completeness={completeness.mean():.1%}, Anomalies={len(anomalies)}")

def calibrate_sensor_pair(df_source, df_target, uids):
    params, quality = {}, {}
    
    for uid in uids:
        merged = pd.merge(
            df_source[df_source['UID'] == uid][['yearmonth', 'NDVI']],
            df_target[df_target['UID'] == uid][['yearmonth', 'NDVI']],
            on='yearmonth', suffixes=('_src', '_tgt'))
        
        if len(merged) < MIN_HARMONIZATION_SAMPLES:
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
        logger.info(f"Global AVHRR→MODIS: slope={global_params[0]:.4f}, intercept={global_params[1]:.4f}")
        logger.info(f"Per-polygon: {len(params)}/{len(uids)} calibrated")
    
    return params, quality, global_params

# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = Path(DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine date range
    start_year, start_month = START_DATE
    end_year, end_month = END_DATE if END_DATE else get_latest_modis_month()
    
    # Check missing months
    months = list_months(start_year, start_month, end_year, end_month)
    missing_months = []
    for y, m in months:
        ym = f"{y:04d}{m:02d}"
        month_csvs = list(out_dir.glob(f"NDVI_*_{ym}.csv"))
        if not month_csvs:
            missing_months.append((y, m))
    
    if not missing_months:
        logger.info("✓ All requested months have CSVs - processing...")
        if consolidate_monthly_csvs(out_dir):
            harmonize_sensors(out_dir)
            logger.info("\n" + "="*60)
            logger.info("✓ PROCESSING COMPLETE")
            logger.info(f"Output: {out_dir}")
            logger.info("="*60)
            return
    
    # Submit GEE tasks
    logger.info(f"Missing {len(missing_months)} months - submitting GEE export tasks...")
    
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
    
    if gee_task_watchdog.launch_watchdog():
        logger.info("✓ Task watchdog launched in new window")
    
    tasks = export_monthly_ndvi(missing_months, polygons)
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS:")
    logger.info("1. Monitor tasks: https://code.earthengine.google.com/tasks")
    logger.info("2. Download CSVs from Google Drive: GEE_AVHRR_MODIS_NDVI")
    logger.info(f"3. Save CSVs to: {out_dir.absolute()}")
    logger.info("4. Re-run this script to harmonize")
    logger.info("="*60)

if __name__ == "__main__":
    main()
