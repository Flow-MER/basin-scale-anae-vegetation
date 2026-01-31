# Basin-scale ANAE Vegetation Vulnerability

Modified from the MDBA BWS Vulnerabilities Project

The BWS Priorities Project aimed to spatially and temporally summarise metrics of vulnerability (combining condition and stress) for vegetation and waterbirds in the Murray-Darling Basin with the aim of informing the setting of annual watering priorities for these target groups.

**Project report**: Hale, J., Brooks, S., Campbell, C. and McGinness, H. (2023) Assessing Vulnerability for use in Determining Basin-scale Environmental Watering Priorities. A Report to the Commonwealth Environmental Water Office, Canberra.

* [LInk to report from the DCCEEW website](https://www.dcceew.gov.au/sites/default/files/documents/assessing-vulnerability-use-determining-basin-scale-environmental-watering-priorities.pdf)

The Jupyter notebook is the final stage of data processing that pulls together multiple data sets to summarise and score the condition metrics, stress metrics and then add the scores to the final vulnerability metric.  Multiple input data files are read in, pivoted to tabular format with years as columns.  The measurement of vulnerability relies on first calculating the long-term baseline (mean of all years excluding the millennium drought) then scoring the deviation from the baseline.   Metrics calculated for ANAE ecosystem polygons are aggregated together as an area weighted average for larger spatial units (e.g. Ramsar sites, valleys).

## Data Inputs

 1. Australian National Aquatic Ecosystem (ANAE) mapping  v3 - The ANAE identifies different vegetation types and provides the spatial units used to summarise other data. Polygons < 1 Ha are removed as they are too small to meet the reliability requirements of the WIT tool and MODIS derived NDVI.  
 1. Geoscience Australia Wetland Insights Tool (WIT) - WIT data observations for all ANAE polygons > 1 Ha in the MDB 1986-present.  Raw data supplied by Geoscience Australia for individual observation dates through the Landsat Record summarised into daily, yearly, all-time and inundation event statistics (a separate jupyter notebook)
 1. Normalized Difference Vegetation Index (NDVI) - Average NDVI per ANAE polygon per year 1986-present calculated using google earth engine reducer: shared code: [(Flow-MER GoogleEarthEngine_scripts)](https://github.com/Flow-MER/GoogleEarthEngine_scripts)
 1. [Root Zone Soil Moisture (Australian Water Outlook)](https://awo.bom.gov.au/products/historical/soilMoisture-rootZone) - Mean root zone soil moisture per ANAE polygon per year was generated using ArcGIS but there are many ways to calculate the annual average per polygon from the AWO netcdf
 1. Stress thresholds for vegetation based on durations since last inundation for different functional groups that were identified by experts are coded directly into this Jupyter Notebook

## Data Outputs

This notebook writes the various metric to the working directory in tabular format csv files (spatial units in rows, years in columns) that can be read by Microsoft Excel.  Baseline values and scores are added to the tables as additional columns. The output includes spatial scales that were not included in the BWS Vulnerabilities project report but may be useful for other investigations or to inform water planning at those locations (e.g. DIWA and Ramsar sites)

Output files for habitat metrics follow the naming convention: {metric}_{aggregator}_{year_window_width}yr_condition.csv
e.g.  pv_median_DIWA_5yr_condition.csv  is the median "pv" (green fractional cover) with ANAE polygons aggregated to larger DIWA wetland scales using a 5-year moving window in which to calculate rates of change.  

*NOTE:  The outputs generated from this notebook will vary from the previous work in the BWS Vulnerabilities report because:

1. removed the MDBA Stand Condition tool inputs
2. threshold NDVI inputs to positive values only (limits influence of areas of open water)
3. removed unvegetated ANAE classes (lakes, clay pans)

### WIT Threshold - time since last inundation
To quantify inundation and intervening dry periods for each wetland polygon, we derived a feature-specific inundation threshold from the long-term distribution of water extent. Monthly fractional inundation (expressed as the proportion of the polygon classified as open water or saturated substrate) was available for a 40-year period for each feature.

For each polygon, an adaptive inundation threshold was defined as the 30th percentile (P30) of the long-term distribution of fractional inundation. This percentile-based approach provides a robust, non-parametric estimate of a characteristic wet condition while limiting sensitivity to extreme wet years, long inundated plateaus, and skewed or zero-inflated distributions commonly observed across heterogeneous wetland types. The use of a lower percentile intentionally biases the threshold toward conservative identification of ecologically meaningful re-wetting events, reflecting the assumption that false positive inundation detections are more detrimental to downstream vegetation stress estimates than delayed detection of inundation.

To prevent spurious classification driven by noise or permanently inundated features, the adaptive threshold was constrained within fixed bounds. A minimum threshold of 0.05 (5% area) was imposed to exclude classification noise and trivial wetting in predominantly dry systems, while a maximum threshold of 0.5 was applied to prevent permanent or near-permanent water bodies from being classified as dry when below average water levels still cover the majority of the area. These bounds ensure consistency of inundation detection across ephemeral, seasonal, and perennial systems.

A Time since last inundation (TSLI) metric is defined as the number of consecutive days since the most recent inundation event. TSLI was updated using a moving temporal window across the full time series, resetting to zero only when fractional inundation exceeded the threshold. Periods below the threshold increment TSLI monotonically, representing accumulating dry duration relevant to vegetation stress.

To assess the sensitivity of TSLI to threshold selection, additional thresholds based on the 20th and 40th percentiles (P20 and P40) were also computed for all features. These alternative thresholds represent more conservative and more permissive inundation definitions, respectively, and provide bounds on uncertainty associated with threshold choice. Sensitivity analyses focused on the effects of threshold variation on derived TSLI metrics, rather than on inundation frequency alone, reflecting the primary role of inundation events as resets of hydrologic memory within vegetation stress modeling.



### Mapping the outputs

* Patterns can be visualised in GIS by joining the output files to the relevant spatial layers.  Many of the vegetation maps in the report used the ANAE polygons scale to visualise the patterns - this was done by joining **FINAL_BWSVulnerability_vegetation_ANAE.csv** to the **ANAEv3** using the **UID** polygon identifier.  Mapping whole Valley aggregated scores would be done by joining **FINAL_BWSVulnerability_vegetation_Valley.csv** to **BWSRegions.shp** using the **BWS_Region**.

## Processing Environment

Python 3.11.11

install requirements

```pip install -r requirements.txt```

## Repeating or extending the analysis to additional years of data

Extending the analysis requires:

1. collating new input data and appending to the current 1986-2024 source files
1. edit the definition of the **alltime** variable to extend past 2024.
1. re-run the notebook

Source data comes from a variety of places and requires a different technologies to assemble as outlined above.  The current source files should be used as the template to append to,  which should ensure the updated files will run with this workbook.  There is some additional code built into the workbook to re-build spatial relationships among data

The code was built to test the method within the confines of a project so it isn't always pretty.    If the logic is not clear please refer to the report and reach out to the report authors with questions.

***

## Contact

Dr Shane Brooks
<https://brooks.eco>

![Brooks.eco logo](brooks-logo.png "Brooks Ecology & Technology")
