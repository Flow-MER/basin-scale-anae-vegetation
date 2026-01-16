
import os
import sys
import arcpy
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import xarray as xr
from config import soil_moisture_cfg as config

arcpy.env.overwriteOutput = True

def build_year_tasks(nc_path, start_year, end_year, time_var="time"):
    # open dataset
    ds = xr.open_dataset(nc_path)
    
    # extract time dimension as pandas datetime index
    times = pd.to_datetime(ds[time_var].values)
    
    # figure out min and max years
    min_year = max(start_year,times.min().year)
    max_year = min(end_year, times.max().year)
    
    tasks = []
    
    for year in range(min_year, max_year + 1):
        # all times in this year
        year_times = times[times.year == year]
        if len(year_times) == 0:
            continue  # skip empty years (shouldn’t happen)
        
        # start = first timestamp in that year
        start = year_times.min().strftime("%Y-%m-%dT%H:%M:%S")
        # end = last timestamp in that year
        end   = year_times.max().strftime("%Y-%m-%dT%H:%M:%S")
        
        tasks.append((year, start, end))
    
    return tasks

def process_year(year, start_date, end_date, zonefile):
    """Process a single year of soil moisture data"""
    crf_path = rf"D:\BWSVulnerability\climate\sm_pct_relative_monthly_{year}.crf"
    result_table = os.path.join(arcpy.env.scratchFolder,f"ZonalSt30_soilmoistureanomally_monthly_{year}.dbf")
    csv_path = os.path.join(arcpy.env.scratchFolder,f"zonal_soilmoisture{year}.csv")
    
    extent = arcpy.Describe(zonefile).extent
    spatialRef = arcpy.Describe(zonefile).spatialReference
    
    with arcpy.EnvManager(outputCoordinateSystem=spatialRef, extent=extent, cellSize=zonefile):
        # Subset multidimensional raster
        arcpy.md.SubsetMultidimensionalRaster(
            in_multidimensional_raster=config.ROOT_ZONE_SOIL_MOISTURE_RELATIVE,
            out_multidimensional_raster=crf_path,
            variables=config.SM_VAR,
            dimension_def="BY_RANGES",
            dimension_ranges=f"StdTime {start_date} {end_date}"
        )
        
        # Zonal statistics
        arcpy.ia.ZonalStatisticsAsTable(
            in_zone_data=zonefile,
            zone_field=config.POLY_UID,
            in_value_raster=crf_path,
            out_table=result_table,
            ignore_nodata="DATA",
            statistics_type="MEAN",
            process_as_multidimensional="ALL_SLICES"
        )
        
        arcpy.conversion.ExportTable(
            in_table=result_table,
            out_table=csv_path,
            field_mapping=f'UID "UID" true true false 9 Text 0 0,First,#,{result_table},UID,0,8;date "date" true true false 8 Date 0 0,First,#,{result_table},StdTime,-1,-1;sm_pct "sm_pct" true true false 8 Double 0 0,First,#,{result_table},MEAN,-1,-1'
        )
        arcpy.management.Delete(result_table)
    return csv_path

def merge_csv(file_list, out_file):
    # read all CSVs into dataframes
    dfs = [pd.read_csv(f) for f in file_list]
    merged = pd.concat(dfs, ignore_index=True)
    merged['date'] = pd.to_datetime(merged['date']).dt.date
    merged['sm_pct'] = merged['sm_pct'].round(4)
    merged.to_csv(out_file, index=False, compression="zip")

if __name__ == "__main__":
    # path to soil moisture netcdf
    nc_path = r"sm_pct_relative_monthly.nc"
    
    crs = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    cellSize = 0.00025   #25
    zonefile = r"ANAEv3_WIT_PolygonToRaster4326"
    extent = f'138.5 -37.7 152.5 -24.5 {crs}'
    out_soil_moisture_monthly_zip = r"D:\FlowMER2.0\basin-scale-anae-vegetation-vulnerability\input\csv\ANAE_soilmoistureanomally_monthly.csv.zip"
    
    
    # Create zone raster once
    if not arcpy.Exists(zonefile):
        with arcpy.EnvManager(outputCoordinateSystem=crs, extent=extent, cellSize=cellSize):
            arcpy.conversion.PolygonToRaster(
                in_features=config.POLYGON_PATH,  
                value_field=config.POLY_UID,
                out_rasterdataset=zonefile,
                cell_assignment="CELL_CENTER",
                cellsize=cellSize,
                build_rat="BUILD"
            )
            
           
    start_year = 2024
    end_year = 2025
    tasks = build_year_tasks(nc_path, start_year, end_year)
    tasks = [(year, start, end, zonefile) for year, start, end in tasks]
    
    results = [os.path.join(arcpy.env.scratchFolder,f"zonal_soilmoisture{year}.csv") for year, start, end, zonefile in tasks]

    #Process years in parallel limiting processes to prevent memory blowout
    with Pool(processes=4) as pool:
        results = pool.starmap(process_year, tasks)
    
    print(f"Processed {len(results)} years successfully")
    
    try:
        os.remove(out_soil_moisture_monthly_zip)
    except OSError:
        pass
    merge_csv(results, out_soil_moisture_monthly_zip)
    if os.path.exists(out_soil_moisture_monthly_zip):
        for f in results:
            try:
                os.remove(f)
            except OSError:
                pass
    else:
        print("Failed to create zip file")
