import os
import pandas as pd
import geopandas as gpd
import xarray as xr
#import rioxarray
from exactextract import exact_extract
from tqdm import tqdm
import zipfile

def process_soil_moisture_exact(
    start_date,
    end_date,
    zones_path,
    nc_path,
    var="sm_pct",
    uid_field="UID",
    out_path="./output/",
    crs_fallback="EPSG:4326",
    chunk_size=12,
):
    """
    Compute monthly zonal statistics from a NetCDF raster time series.
    
    Optimizations:
    - Batch processing in chunks to reduce I/O
    - Memory-efficient data handling
    - Vectorized date operations
    """
    os.makedirs(out_path, exist_ok=True)
    
    # Load and validate zones
    zones = gpd.read_file(zones_path)
    if zones.crs is None:
        raise ValueError("Zones file must have a defined CRS")

    # Open dataset with chunking for memory efficiency
    with xr.open_dataset(nc_path, chunks={'time': chunk_size}) as ds:
        ds = ds.sel(time=slice(start_date, end_date))
        
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in dataset.")
        da = ds[var]

        # Optimize dimension handling
        dim_mapping = {}
        if "longitude" in da.dims:
            dim_mapping["longitude"] = "x"
        if "latitude" in da.dims:
            dim_mapping["latitude"] = "y"
        if dim_mapping:
            da = da.rename(dim_mapping)

        # Set CRS and reproject zones
        if not da.rio.crs:
            da = da.rio.write_crs(crs_fallback)
        zones = zones.to_crs(da.rio.crs)

        # Clip to bounds
        da = da.rio.clip_box(*zones.total_bounds)

        # Process in chunks and collect results
        output_file = os.path.join(out_path, f"soil_moisture_zonal_{start_date}_{end_date}.csv")
        results = []
        
        for i in tqdm(range(len(da.time)), desc="Processing time slices"):
            slice_da = da.isel(time=i)
            stats = exact_extract(slice_da, zones, "mean", include_cols=[uid_field], output="pandas")
            stats["date"] = pd.to_datetime(da.time.values[i]).replace(day=1)
            results.append(stats)
            
            # Write in chunks to manage memory
            if len(results) >= chunk_size or i == len(da.time) - 1:
                chunk_df = pd.concat(results, ignore_index=True)
                chunk_df.to_csv(output_file, mode="a", header=(i < chunk_size), index=False)
                results = []

    return output_file


if __name__ == "__main__":
    zones_file = r"D:\BWSVulnerability\WIT\ANAEv3_WIT_clean16052025\ANAEv3_WIT.shp"
    nc_file = r"D:\BWSVulnerability\climate\sm_pct_relative_monthly.nc"

    csv_path = process_soil_moisture_exact(
        start_date="1987-01",
        end_date="1988-01",
        zones_path=zones_file,
        nc_path=nc_file,
        var="sm_pct",
        uid_field="UID",
        out_path="./output/",
        crs_fallback="EPSG:4326",
        chunk_size=12,
    )

    print(f"Output saved to: {csv_path}")
    
    # Compress output
    zip_file = csv_path + ".zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(csv_path, os.path.basename(csv_path))
    print(f"Compressed to: {zip_file}")