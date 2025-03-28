# Basin Scale Vegetation Vulnerability

Modified from the MDBA BWS Vulnerabilities Project

## Data Inputs

 1. Australian National Aquatic Ecosystem (ANAE) mapping  v3 - The ANAE identifies different vegetation types and provides the spatial units used to summarise other data. Polygons < 1 Ha are removed as they are too small to meet the reliability requirements of the WIT tool and MODIS derived NDVI.  
 1. Geosciences Australia Wetland Insights Tool (WIT) - WIT data observations for all ANAE polygons > 1 Ha in the MDB 1986-present.  Raw data supplied by Geosciences Australia for individual observation dates through the Landsat Record summarised into daily, yearly, all-time and inundation event statistics (a separate jupyter notebook)
 1. Normalized Difference Vegetation Index (NDVI) - Average NDVI per ANAE polygon per year 1986-present calculated using google earth engine reducer: shared code: <https://github.com/Flow-MER/GoogleEarthEngine_scripts>
 1. Root Zone Soil Moisture (Australian Water Outlook) - Mean root zone soil moisture per ANAE polygon per year was generated using ArcGIS but there are many ways to calculate the annual average per polygon from the AWO netcdf   <https://awo.bom.gov.au/products/historical/soilMoisture-rootZone>
 1. Stress thresholds for vegetation based on durations since last inundation for different functional groups that were identified by experts are coded directly into this Jupyter Notebook

## Data Outputs

This notebook writes the various metric to the working directory in tabular format csv files (spatial units in rows, years in columns) that can be read by Microsoft Excel.  Baseline values and scores are added to the tables as additional columns. There are a **lot** of output files included for spatial scales that were not included in the project report but may be useful for other investigations or to inform water planning at those locations (e.g. DIWA and Ramsar sites)

Output files for habitat metrics follow the naming convention: {metric}_{aggregator}_{year_window_width}yr_condition.csv
e.g.  pv_median_DIWA_5yr_condition.csv  is the median "pv" (green fractional cover) with ANAE polygons aggregated to larger DIWA wetland scales using a 5-year moving window in which to calculate rates of change.  

*NOTE:  The outputs generated from this notebook will vary from the report because this code has removed the MDBA Stand Condition tool inputs and made improvements to the NDVI inputs

### Mapping the outputs

* Patterns can be visualised in GIS by joining the output files to the relevant spatial layers.  Many of the vegetation maps in the report used the ANAE polygons scale to visualise the patterns - this was done by joining **FINAL_BWSVulnerability_vegetation_ANAE.csv** to the **ANAEv3** using the **UID** polygon identifier.  Mapping whole Valley aggregated scores would be done by joining **FINAL_BWSVulnerability_vegetation_Valley.csv** to **BWSRegions.shp** using the **BWS_Region**.

## Processing Environment

For the project the analysis was conducted in the python processing environment of ArcGIS Pro 3.0 but were coded to use common open source python data processing libraries (Geopandas, Pandas, numpy) that should enable the analysis to be repeated in most environments.
Note the code produces many more tables of output data than are required because it calculates parameters for multiple spatial scales.  The BWS Vulnerability Project reports mainly on vegetation outcomes at the scale of MDB Valleys (BWS Vegetation Regions). Outputs are also generated for Ramsar sites, MDBA Waterbird Areas (the "dirty thirty" polygons), and individual ANAE polygons.

## Repeating or extending the analysis to additional years of data

Extending the analysis requires:

1. collating new data and appending to the current 1986-2024 source files
1. edit the definition of the **alltime** variable to extend past 2024.
1. re-run the notebook

Source data comes from a variety of places and requires a different technologies to assemble as outlined above.  The current source files should be used as the template to append to,  which should ensure the updated files will run with this workbook.  There is some additional code built into the workbook to re-build spatial relationships among data

The code was built to test the method within the confines of a project so it isn't always pretty.    If the logic is not clear please refer to the report and reach out to the report authors with questions.

***

# Contact

* Dr Shane Brooks

* <https://brooks.eco>
