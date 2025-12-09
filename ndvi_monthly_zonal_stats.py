"""
Monthly NDVI Zonal Statistics Extraction

Extracts monthly NDVI for ANAE polygons using Google Earth Engine with robust
cross-sensor calibration between AVHRR (1986-2013) and MODIS (2000-present).

Key Features:
- Automatic detection of latest available MODIS data
- Quality filtering using sensor QA bands (AVHRR QA=0, MODIS SummaryQA≤1)
- Robust Huber regression for AVHRR-MODIS calibration (resistant to outliers)
- Incremental processing: only extracts new months on subsequent runs
- Metadata tracking: standard deviation, pixel counts per polygon
- Quality control reports: temporal completeness, anomalous jumps

Output Files (annual CSVs):
- NDVI_{sensor}_{year}.csv: Raw sensor data with metadata
- NDVI_recalibrated_{year}.csv: Cross-calibrated timeseries
- calibration_quality_metrics.csv: R², RMSE, sample size per polygon
- temporal_completeness.csv: Data availability per polygon
- anomalous_jumps.csv: Month-to-month changes >30%

Usage:
    python ndvi_monthly_zonal_stats.py
    # Edit main() parameters: polygon_file, asset_id, incremental=True/False

Author: Shane Brooks (brooks.eco)
Modified: 2025-01-08
"""

import ee
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
import geopandas as gpd
import geemap
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Initialize GEE
# ----------------------------
ee.Initialize()

# Configuration
DEFAULT_OUT_DIR = "./ndvi_output"  # Output directory for CSV files
AVHRR_YEARS = (1986, 2013)  # AVHRR CDR v5 temporal coverage
OVERLAP_YEARS = (2000, 2013)  # Calibration period (both sensors available)
SCALE = 0.0001  # NDVI scale factor for both sensors
REDUCE_SCALE = 30  # Spatial resolution (meters) for zonal statistics
MIN_CALIBRATION_SAMPLES = 24  # Min overlap months for per-polygon calibration

# ----------------------------
# Polygon upload / handling
# ----------------------------
def upload_polygons_to_gee(local_file: str, asset_id: str) -> ee.FeatureCollection:
    """Load polygons from GEE asset or upload local file.
    
    Args:
        local_file: Path to .shp or .geojson file
        asset_id: GEE asset ID (e.g., 'users/username/asset_name')
    
    Returns:
        ee.FeatureCollection with UID field for polygon identification
    """
    try:
        fc = ee.FeatureCollection(asset_id)
        logger.info("Using existing GEE asset: %s", asset_id)
        return fc
    except ee.ee_exception.EEException:
        logger.info("Uploading local polygon file to GEE: %s", local_file)
        local_path = Path(local_file)
        ext = local_path.suffix.lower()
        if ext in [".shp", ".geojson"]:
            fc = geemap.shp_to_ee(str(local_file), asset_id=asset_id)
        else:
            raise ValueError("Unsupported polygon file type. Must be .shp or .geojson")
        logger.info("Upload complete: %s", asset_id)
        return ee.FeatureCollection(asset_id)

# ----------------------------
# Helper functions
# ----------------------------
def list_months(start_year: int, end_year: int):
    """Generate list of (year, month) tuples for date range."""
    return [(y, m) for y in range(start_year, end_year+1) for m in range(1, 13)]

def month_start_end(year: int, month: int):
    """Get GEE Date objects for month boundaries (inclusive start, exclusive end)."""
    start = ee.Date.fromYMD(year, month, 1)
    end = ee.Date.fromYMD(year + int(month==12), month % 12 + 1, 1)
    return start, end

def get_modis_latest_complete_month():
    """Detect most recent complete month of MODIS MOD13Q1 data.
    
    Checks last 3 months to find most recent month with available imagery.
    MODIS data typically has 1-2 month latency.
    
    Returns:
        Tuple of (year, month) for latest complete month
    """
    now = datetime.now()
    for months_back in range(1, 4):
        check_date = now - timedelta(days=30*months_back)
        year, month = check_date.year, check_date.month
        start = ee.Date.fromYMD(year, month, 1)
        end = ee.Date.fromYMD(year + int(month==12), month % 12 + 1, 1)
        coll = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end)
        count = coll.size().getInfo()
        if count > 0:
            logger.info(f"Latest complete MODIS month: {year}-{month:02d}")
            return year, month
    logger.warning("Could not detect recent MODIS data, using current year")
    return now.year, now.month

def reduce_regions_monthly(img_collection, polygons):
    """Compute zonal statistics (mean, std dev, count) per polygon.
    
    Args:
        img_collection: ee.ImageCollection with 'yearmonth' property
        polygons: ee.FeatureCollection with 'UID' field
    
    Returns:
        Flattened ee.FeatureCollection with NDVI, NDVI_sd, pixel_count per polygon
    """
    def reducer_func(img):
        stats = img.reduceRegions(
            collection=polygons,
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), '', True
            ).combine(
                ee.Reducer.count(), '', True
            ),
            scale=REDUCE_SCALE,
            tileScale=4
        ).map(lambda f: f.select(
            ['mean','stdDev','count','UID'], 
            ['NDVI','NDVI_sd','pixel_count','UID']
        ).set({
            'yearmonth': img.get('yearmonth'),
            'date_start': img.get('system:time_start')
        }))
        return stats.filter(ee.Filter.notNull(['NDVI']))
    return img_collection.map(reducer_func).flatten()

# ----------------------------
# Build monthly collections
# ----------------------------
def build_monthly_collection(sensor: str, polygons, aoi, start_year: int, end_year: int, end_month: int = 12):
    """Build monthly composites with quality filtering for full extraction.
    
    Quality filtering:
    - AVHRR: QA == 0 (good quality only)
    - MODIS: SummaryQA <= 1 (good/marginal), NDVI > 0 (remove water/clouds)
    
    Args:
        sensor: 'AVHRR' or 'MODIS'
        polygons: ee.FeatureCollection (unused but kept for API consistency)
        aoi: ee.Geometry for clipping
        start_year: First year to extract
        end_year: Last year to extract
        end_month: Last month to extract in end_year (default 12)
    
    Returns:
        ee.ImageCollection with monthly mean NDVI composites
    """
    imgs = []
    months = list_months(start_year, end_year)
    months = [(y,m) for y,m in months if y < end_year or m <= end_month]
    
    for y,m in months:
        start, end = month_start_end(y,m)
        if sensor.lower() == 'avhrr':
            coll = ee.ImageCollection("NOAA/CDR/AVHRR/NDVI/V5").filterDate(start,end)
            coll = coll.map(lambda img: img.updateMask(img.select('QA').eq(0)))
            img = coll.select("NDVI").mean().multiply(SCALE).clip(aoi)
        elif sensor.lower() == 'modis':
            coll = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start,end)
            coll = coll.map(lambda img: img.updateMask(img.select('SummaryQA').lte(1)))
            img = coll.select("NDVI").mean().multiply(SCALE).clip(aoi)
            img = img.updateMask(img.gt(0))
        else:
            raise ValueError("Sensor must be AVHRR or MODIS")
        img = img.set('yearmonth', f"{y:04d}{m:02d}")
        imgs.append(img)
    return ee.ImageCollection(imgs)

# ----------------------------
# Incremental extraction helpers
# ----------------------------
def get_last_extracted_month(out_dir: str, sensor: str):
    """Find most recent month in existing CSV files for incremental extraction.
    
    Returns:
        Integer YYYYMM or None if no existing files
    """
    out_dir = Path(out_dir)
    csv_files = list(out_dir.glob(f"NDVI_{sensor}_*.csv"))
    if not csv_files:
        return None
    max_ym = 0
    for f in csv_files:
        df = pd.read_csv(f)
        if not df.empty:
            max_ym = max(max_ym, df['yearmonth'].max())
    return max_ym

def build_incremental_collection(sensor: str, polygons, aoi, out_dir: str, start_year: int, end_year: int, end_month: int = 12):
    """Build collection of only new months since last extraction.
    
    Checks existing CSV files to determine last extracted month, then builds
    collection for subsequent months only. Returns empty collection if no new data.
    
    Args:
        sensor: 'AVHRR' or 'MODIS'
        polygons: ee.FeatureCollection (unused but kept for API consistency)
        aoi: ee.Geometry for clipping
        out_dir: Directory containing existing CSV files
        start_year: First year of sensor coverage
        end_year: Last year to check for new data
        end_month: Last month to check in end_year
    
    Returns:
        ee.ImageCollection with only new monthly composites
    """
    last_ym = get_last_extracted_month(out_dir, sensor)
    months = list_months(start_year, end_year)
    months = [(y,m) for y,m in months if y < end_year or m <= end_month]
    
    if last_ym is not None:
        months = [(y,m) for y,m in months if y*100 + m > last_ym]
        logger.info("%s incremental extraction: %d new months", sensor, len(months))
    else:
        logger.info("%s full extraction: %d months", sensor, len(months))
    
    if not months:
        logger.info("%s: No new months to extract", sensor)
        return ee.ImageCollection([])
    
    imgs = []
    for y,m in months:
        start, end = month_start_end(y,m)
        if sensor.lower() == 'avhrr':
            coll = ee.ImageCollection("NOAA/CDR/AVHRR/NDVI/V5").filterDate(start,end)
            coll = coll.map(lambda img: img.updateMask(img.select('QA').eq(0)))
            img = coll.select("NDVI").mean().multiply(SCALE).clip(aoi)
        elif sensor.lower() == 'modis':
            coll = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start,end)
            coll = coll.map(lambda img: img.updateMask(img.select('SummaryQA').lte(1)))
            img = coll.select("NDVI").mean().multiply(SCALE).clip(aoi)
            img = img.updateMask(img.gt(0))
        else:
            raise ValueError("Sensor must be AVHRR or MODIS")
        img = img.set('yearmonth', f"{y:04d}{m:02d}")
        imgs.append(img)
    return ee.ImageCollection(imgs)

def append_new_months_to_csv(df_new: pd.DataFrame, out_dir: str, sensor: str):
    """Append new monthly data to annual CSV files, removing duplicates.
    
    Creates new annual CSV if it doesn't exist, otherwise merges with existing.
    Ensures no duplicate UID-yearmonth combinations.
    """
    out_dir = Path(out_dir)
    for year in df_new['yearmonth']//100:
        df_year_new = df_new[(df_new['yearmonth']//100) == year]
        out_file = out_dir / f"NDVI_{sensor}_{year}.csv"
        if out_file.exists():
            df_existing = pd.read_csv(out_file)
            df_combined = pd.concat([df_existing, df_year_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['UID','yearmonth'], inplace=True)
        else:
            df_combined = df_year_new
        df_combined.to_csv(out_file, index=False)
        logger.info("Updated CSV: %s", out_file)

# ----------------------------
# Throttled parallel export
# ----------------------------
def export_collection_to_csv_throttled(img_collection, polygons, out_dir: str, sensor: str,
                                       max_workers: int = 2, throttle_sec: float = 10.0):
    """Extract zonal statistics and save to annual CSV files.
    
    Processes entire collection in one GEE call, then splits results by year.
    Parameters max_workers and throttle_sec are legacy (not currently used).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reducing image collection to polygons...")
    zonal_stats = reduce_regions_monthly(img_collection, polygons)
    features = zonal_stats.getInfo()['features']
    if not features:
        logger.info("No new features for %s", sensor)
        return
    df = pd.DataFrame([{
        'UID': f['properties']['UID'],
        'yearmonth': int(f['properties']['yearmonth']),
        'date_start': f['properties']['date_start'],
        'NDVI': f['properties']['NDVI'],
        'NDVI_sd': f['properties'].get('NDVI_sd', np.nan),
        'pixel_count': f['properties'].get('pixel_count', np.nan),
        'sensor': sensor
    } for f in features])
    append_new_months_to_csv(df, out_dir, sensor)

# ----------------------------
# Recalibrate AVHRR
# ----------------------------
def recalibrate_avhrr(local_out_dir: str, overlap_start: int = 2000, overlap_end: int = 2013):
    """Calibrate AVHRR to MODIS scale using Huber regression on overlap period.
    
    Method:
    1. Per-polygon calibration where ≥24 overlap months exist
    2. Global calibration as fallback for polygons with insufficient data
    3. Huber regression (epsilon=1.35) for robustness to outliers
    4. Quality metrics (R², RMSE) saved for validation
    
    Only AVHRR data before overlap_start is recalibrated; MODIS data unchanged.
    
    Args:
        local_out_dir: Directory containing NDVI_{sensor}_{year}.csv files
        overlap_start: First year of calibration period (default 2000)
        overlap_end: Last year of calibration period (default 2013)
    
    Outputs:
        - NDVI_recalibrated_{year}.csv: Combined calibrated timeseries
        - calibration_quality_metrics.csv: R², RMSE, n per polygon
        - temporal_completeness.csv: Data availability per polygon
        - anomalous_jumps.csv: Month-to-month changes >30%
    """
    logger.info("Starting robust AVHRR recalibration using MODIS overlap period...")
    csv_files = list(Path(local_out_dir).glob("NDVI_*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", local_out_dir)
        return
    
    df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df_all['yearmonth'] = df_all['yearmonth'].astype(int)
    
    df_overlap = df_all[
        (df_all['yearmonth'] >= overlap_start*100+1) & 
        (df_all['yearmonth'] <= overlap_end*100+12)
    ]
    df_pivot = df_overlap.pivot_table(
        index=['UID','yearmonth'], columns='sensor', values='NDVI'
    ).reset_index()
    df_pivot.dropna(subset=['AVHRR','MODIS'], inplace=True)
    
    calibration_params = {}
    calibration_quality = {}
    
    # Per-polygon calibration
    for uid, grp in df_pivot.groupby('UID'):
        if len(grp) < MIN_CALIBRATION_SAMPLES:
            logger.warning(f"UID {uid}: insufficient samples ({len(grp)}), using global calibration")
            continue
        
        X = grp['AVHRR'].values.reshape(-1,1)
        y = grp['MODIS'].values
        
        # Huber regression (robust to outliers)
        model = HuberRegressor(epsilon=1.35).fit(X, y)
        
        # Quality metrics
        y_pred = model.predict(X)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        
        calibration_params[uid] = (model.coef_[0], model.intercept_)
        calibration_quality[uid] = {'r2': r2, 'rmse': rmse, 'n': len(grp)}
        
        if r2 < 0.5:
            logger.warning(f"UID {uid}: poor calibration (R²={r2:.3f})")
    
    # Global fallback calibration
    X_global = df_pivot['AVHRR'].values.reshape(-1,1)
    y_global = df_pivot['MODIS'].values
    global_model = HuberRegressor(epsilon=1.35).fit(X_global, y_global)
    global_params = (global_model.coef_[0], global_model.intercept_)
    logger.info(f"Global calibration: slope={global_params[0]:.4f}, intercept={global_params[1]:.4f}")
    
    # Apply calibration
    df_all['NDVI_recalibrated'] = df_all['NDVI']
    mask_avhrr = (df_all['yearmonth'] < overlap_start*100+1)
    
    for uid in df_all['UID'].unique():
        params = calibration_params.get(uid, global_params)
        a, b = params
        mask_uid = mask_avhrr & (df_all['UID'] == uid)
        df_all.loc[mask_uid, 'NDVI_recalibrated'] = df_all.loc[mask_uid, 'NDVI']*a + b
    
    # Save calibration metadata
    if calibration_quality:
        pd.DataFrame(calibration_quality).T.to_csv(
            Path(local_out_dir)/"calibration_quality_metrics.csv"
        )
        logger.info("Calibration quality metrics saved")
    
    # Validation metrics
    validate_timeseries_quality(df_all, local_out_dir)
    
    # Save recalibrated data
    for year in sorted(df_all['yearmonth']//100):
        df_year = df_all[(df_all['yearmonth']//100) == year]
        out_file = Path(local_out_dir)/f"NDVI_recalibrated_{year}.csv"
        df_year.to_csv(out_file, index=False)
    
    logger.info("Robust AVHRR recalibration completed.")

def validate_timeseries_quality(df: pd.DataFrame, out_dir: str):
    """Generate QC reports for temporal completeness and anomalous jumps.
    
    Completeness: Fraction of expected months with data per polygon
    Anomalies: Month-to-month NDVI changes >0.3 (30% absolute change)
    """
    logger.info("Generating QC reports...")
    
    # Temporal completeness
    min_year = df['yearmonth'].min() // 100
    max_year = df['yearmonth'].max() // 100
    expected_months = set((y,m) for y in range(min_year, max_year+1) for m in range(1,13))
    
    completeness = df.groupby('UID').apply(
        lambda x: len(set(zip(x['yearmonth']//100, x['yearmonth']%100))) / len(expected_months)
    )
    completeness.to_csv(Path(out_dir)/"temporal_completeness.csv")
    
    # Detect anomalous jumps
    df_sorted = df.sort_values(['UID','yearmonth'])
    df_sorted['ndvi_diff'] = df_sorted.groupby('UID')['NDVI_recalibrated'].diff()
    anomalies = df_sorted[df_sorted['ndvi_diff'].abs() > 0.3]
    anomalies.to_csv(Path(out_dir)/"anomalous_jumps.csv", index=False)
    
    logger.info(f"QC: Mean completeness = {completeness.mean():.2%}")
    logger.info(f"QC: {len(anomalies)} anomalous jumps detected")

# ----------------------------
# Main workflow
# ----------------------------
def main(polygon_file: str, asset_id: str = "users/your_username/anaev3_BWS", incremental: bool = True):
    """Main extraction workflow with automatic MODIS end date detection.
    
    Workflow:
    1. Load/upload polygons to GEE
    2. Detect latest available MODIS month
    3. Extract AVHRR (1986-2013) - incremental or full
    4. Extract MODIS (2000-latest) - incremental or full
    5. Cross-calibrate AVHRR to MODIS scale
    6. Generate QC reports
    
    Args:
        polygon_file: Local path to .shp or .geojson with UID field
        asset_id: GEE asset ID for polygon storage
        incremental: If True, only extract new months; if False, full extraction
    
    Outputs:
        All CSV files written to DEFAULT_OUT_DIR (./ndvi_output/)
    """
    out_dir = DEFAULT_OUT_DIR
    polygons = upload_polygons_to_gee(polygon_file, asset_id)
    aoi = ee.Geometry.Polygon(
        [[[138.5,-37.6],
          [152.5,-37.6],
          [152.5,-24.5],
          [138.5,-24.5]]], None, False
    )
    
    # Detect latest MODIS data
    modis_end_year, modis_end_month = get_modis_latest_complete_month()

    # AVHRR (fixed period)
    logger.info("Processing AVHRR...")
    if incremental:
        avhrr_coll = build_incremental_collection('AVHRR', polygons, aoi, out_dir, AVHRR_YEARS[0], AVHRR_YEARS[1], 12)
    else:
        avhrr_coll = build_monthly_collection('AVHRR', polygons, aoi, AVHRR_YEARS[0], AVHRR_YEARS[1], 12)
    export_collection_to_csv_throttled(avhrr_coll, polygons, out_dir, 'AVHRR', max_workers=2, throttle_sec=15)

    # MODIS (dynamic end date)
    logger.info(f"Processing MODIS (2000 to {modis_end_year}-{modis_end_month:02d})...")
    if incremental:
        modis_coll = build_incremental_collection('MODIS', polygons, aoi, out_dir, 2000, modis_end_year, modis_end_month)
    else:
        modis_coll = build_monthly_collection('MODIS', polygons, aoi, 2000, modis_end_year, modis_end_month)
    export_collection_to_csv_throttled(modis_coll, polygons, out_dir, 'MODIS', max_workers=2, throttle_sec=15)

    # Recalibrate AVHRR
    recalibrate_avhrr(out_dir)

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Configuration: Update these parameters for your use case
    # incremental=True: Only extract new months (efficient for routine updates)
    # incremental=False: Full extraction from 1986 (use for initial run or reprocessing)
    main(
        polygon_file="anaev3_BWS.shp",
        asset_id="users/your_username/anaev3_BWS",
        incremental=True
    )
