# AVHRR-MODIS NDVI Data Lineage

## Overview
Monthly NDVI time series (1986-present) for 230,000 ANAE wetland polygons (>1ha) in the Murray-Darling Basin, combining AVHRR (1986-2000) and MODIS (2000-present) sensors.
## Data Sources

**AVHRR (1986-2013):** `NOAA/CDR/AVHRR/NDVI/V5` - daily composites at ~5km resolution  
**MODIS (2000-present):** `MODIS/061/MOD13Q1` - 16-day composites at 250m resolution  
**Overlap period:** March 2000 to December 2013 (166 months) used for AVHRR→MODIS calibration

## Processing Workflow

1. **Extraction:** Google Earth Engine processing at 250m resolution within Murray-Darling Basin (138.5°E-152.5°E, 24.5°S-37.6°S). Quality filters remove clouds (AVHRR QA bits 1,3) and poor observations (MODIS SummaryQA bits 0-1 ≤ 1). NDVI < 0 (open water) excluded.

2. **Zonal statistics:** Compute monthly mean NDVI, standard deviation, and pixel count per ANAE polygon.

3. **Sensor selection:** MODIS used directly (2000+). AVHRR harmonized using per-polygon Huber regression (ε=1.35, minimum 24 overlap months) or global regression fallback (<0.1% of polygons).


## Output Data Structure

The harmonized dataset is delivered as annual CSV files in `NDVI_Harmonized_Annual.zip` following the naming convention `NDVI_Harmonized_{YEAR}.csv`. Each record contains six fields:
1. ANAE polygon identifier (UID),
2. yearmonth in YYYYMM format
3. harmonized NDVI value (0-1, four decimal precision)
4. standard deviation (NDVI_sd)
5. pixel count
6. provenance (sensor)

Data provenance tracks processing history through three classifications: `MODIS` (direct use from 2000+), `AVHRR_harmonized` (polygon-specific calibration), and `AVHRR_harmonized_global` (global calibration fallback).

Quality control documentation includes `harmonization_quality.csv` (per-polygon R², RMSE, sample counts), `sensor_summary.csv` (data availability by sensor), `temporal_completeness.csv` (fraction of months with data per polygon), and `anomalous_jumps.csv` (month-to-month NDVI changes >30%).

## Data Quality Considerations

**Strengths:** 39-year continuous time series (1986-present), robust per-polygon calibration minimizing sensor biases, comprehensive quality filtering removing clouds and water pixels, 250m effective resolution suitable for landscape analysis, and coverage of 230,000 wetland polygons across the Murray-Darling Basin.

**Limitations:** Resolution differences between AVHRR (~5km) and MODIS (250m) introduce scale uncertainties for smaller polygons, temporal gaps from cloud cover or sensor availability may affect trend analysis, harmonization uncertainty varies spatially based on overlap data quality, and small polygons are susceptible to edge effects and mixed pixel influences.

## Technical Specifications

Implemented in Python 3.11+ using `earthengine-api`, `pandas`, `scikit-learn`, and `multiprocessing`. Processing combines Google Earth Engine cloud computing for data extraction with local resources for harmonization.


## Version Control and Metadata

**Generated:** 2025-01-10  
**Sources:** ANAE v3 (>1ha polygons), AVHRR v5, MODIS collection 061

## Contact and Attribution

Developed by Shane Brooks (brooks.eco) for Flow-MER Basin-scale ANAE Vegetation Vulnerability project. Cite underlying satellite collections, ANAE v3 mapping, and this processing methodology when using this dataset.