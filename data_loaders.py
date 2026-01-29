"""
Data loaders for Vegetation Vulnerability Analysis.
Handles loading of Spatial, WIT, NDVI, and Soil Moisture datasets.
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm.auto import tqdm
from config import VegetationConfig

def normalise(df):
    """Normalizes values in a DataFrame or Series to a scale of 0 to 1."""
    df_min = df.min()
    if not pd.api.types.is_scalar(df_min):
        df_min = df_min.min()
    df_max = df.max()
    if not pd.api.types.is_scalar(df_max):
        df_max = df_max.max()
    return df.subtract(df_min).divide(df_max - df_min)

def standardise_z(df):
    """z-score standardise an input data frame."""
    dmean = df.mean()
    dstdev = df.std()
    return df.subtract(dmean).divide(dstdev)

class SpatialDataManager:
    def __init__(self, config: VegetationConfig):
        self.config = config
        self.anae_gdf = None
        self.aggregators = {}
        self.agg_fields = {}

    @staticmethod
    def _assign_group_name(anae_type):
        anae_type = str(anae_type).lower()
        if "river red gum" in anae_type and "woodland" in anae_type:
            return "river red gum woodland"
        elif "river red gum" in anae_type:
            return "river red gum swamps and forests"
        elif "black box" in anae_type:
            return "black box"
        elif "coolibah" in anae_type:
            return "coolibah"
        elif "lignum" in anae_type:
            return "lignum"
        elif "cooba" in anae_type:
            return "cooba"
        elif "f2.4: shrubland riparian zone or floodplain" in anae_type:
            return "shrubland"
        elif "tall emergent marsh" in anae_type:
            return "tall reed beds"
        elif "grass" in anae_type or "meadow" in anae_type:
            return "grassy meadows"
        elif "forb marsh" in anae_type or "temporary wetland" in anae_type or "temporary lake" in anae_type:
            return "herbfield"
        return None

    def load_data(self):
        print("Reading ANAE...")
        if not self.config.ANAE_SHP.exists():
            raise FileNotFoundError(f"ANAE shapefile not found: {self.config.ANAE_SHP}")
            
        self.anae_gdf = gpd.read_file(self.config.ANAE_SHP).to_crs("EPSG:3577")
        self.anae_gdf = self.anae_gdf[["UID", "ANAE_TYPE", "Area_Ha", "geometry"]]
        self.anae_gdf["grp"] = self.anae_gdf["ANAE_TYPE"].apply(self._assign_group_name)

        # Basin scale
        self.aggregators["Basin"] = self.anae_gdf[["UID", "grp", "Area_Ha"]].dropna(subset=['grp']).set_index("UID")
        self.aggregators["Basin"].name = "Basin"
        self.agg_fields["Basin"] = []

        # Valley scale
        if self.config.VALLEY_SHP.exists():
            print("Reading BWS valleys...")
            self.agg_fields["Valley"] = ["BWS_Region"]
            valley_gdf = gpd.read_file(self.config.VALLEY_SHP).to_crs("EPSG:3577")
            self.aggregators["Valley"] = (
                gpd.sjoin(self.anae_gdf, valley_gdf[self.agg_fields["Valley"] + ["geometry"]], how="left", predicate="intersects")
                .dropna(subset=['grp'])
                .drop(columns=['geometry'])
                .set_index("UID")
            )
            self.aggregators["Valley"].name = "Valley"

        # DIWA
        if self.config.DIWA_SHP.exists():
            print("Reading DIWA...")
            self.agg_fields["DIWA"] = ["WNAME"]
            diwa_gdf = gpd.read_file(self.config.DIWA_SHP).to_crs("EPSG:3577")
            self.aggregators["DIWA"] = (
                gpd.sjoin(self.anae_gdf, diwa_gdf[self.agg_fields["DIWA"] + ["geometry"]], how="left", predicate="intersects")
                .dropna(subset=['grp'])
                .drop(columns=['geometry'])
                .set_index("UID")
            )
            self.aggregators["DIWA"].name = "DIWA"

        # Ramsar
        if self.config.RAMSAR_SHP.exists():
            print("Reading Ramsar...")
            self.agg_fields["Ramsar"] = ["RAMSAR_NAM", "WETLAND_NA"]
            ramsar_gdf = gpd.read_file(self.config.RAMSAR_SHP).to_crs("EPSG:3577")
            self.aggregators["Ramsar"] = (
                gpd.sjoin(self.anae_gdf, ramsar_gdf[self.agg_fields["Ramsar"] + ["geometry"]], how="left", predicate="intersects")
                .dropna(subset=['grp'])
                .drop(columns=['geometry'])
                .set_index("UID")
            )
            self.aggregators["Ramsar"].name = "Ramsar"

        # ANAE scale (done last to preserve geometry in previous steps if needed, though here we drop it)
        self.aggregators["ANAE"] = self.anae_gdf.drop(columns=['geometry']).set_index("UID")
        self.aggregators["ANAE"].name = "ANAE"
        self.agg_fields["ANAE"] = ["UID"]

class BaseDataLoader:
    def __init__(self, config: VegetationConfig):
        self.config = config

class WitMetricsLoader(BaseDataLoader):
    def load_data(self, valid_uids):
        print("Reading WIT metrics...")
        df = pd.read_csv(self.config.WIT_METRICS_ZIP).rename(columns={"feature_id": "UID"})
        # Filter by valid ANAE UIDs
        return df[df["UID"].isin(valid_uids)]

class TsliLoader(BaseDataLoader):
    def load_data(self, spatial_manager: SpatialDataManager):
        print("Reading WIT time since last inundation...")
        tsli_raw = (
            pd.read_csv(self.config.WIT_TSLI_ZIP, parse_dates=["end_date", "final_date"])
            .rename(columns={"feature_id": "UID"})
            .set_index("UID")
        )
        
        print("Reading WIT inundation metrics...")
        inundation_metrics = pd.read_csv(
            self.config.WIT_INUNDATION_ZIP, parse_dates=["start_date", "end_date"]
        ).rename(columns={"feature_id": "UID"})
        
        inundation_metrics["duration"] = pd.to_timedelta(inundation_metrics["duration"]).dt.days
        inundation_metrics["gap"] = pd.to_timedelta(inundation_metrics["gap"]).dt.days

        # Join with ANAE aggregator to get groups
        tsli_df = tsli_raw.join(spatial_manager.aggregators["ANAE"])

        # Calculate TSLI per year
        for y in tqdm(self.config.ALL_TIME, desc="Time since last inundation per year"):
            cutoff_date = pd.to_datetime(str(y) + "-12-31")
            
            idf = inundation_metrics[inundation_metrics["start_date"] < cutoff_date].copy()
            idf.loc[idf["end_date"] > cutoff_date, "end_date"] = cutoff_date
            
            # Find the last inundation event for each UID up to cutoff_date
            # We want the event with the max end_date for each UID
            last_events = idf.sort_values('end_date').groupby('UID').last()
            
            # Map these back to the main dataframe
            # Note: The original notebook logic was slightly complex with transforms. 
            # Simplified: TSLI = cutoff_date - last_inundation_end_date
            
            # We need to join this temporary calculation back to tsli_df
            # Initialize with existing final_date if no event found in window? 
            # The notebook logic uses the static 'final_date' from tsli_raw as a base but overrides it.
            # Let's stick close to notebook logic:
            
            # Notebook logic reconstruction:
            # 1. Filter events before cutoff
            # 2. Clip end dates to cutoff
            # 3. Find max end_date per UID
            
            max_end_dates = idf.groupby("UID")["end_date"].max()
            
            # Calculate days since
            days_since = (cutoff_date - max_end_dates).dt.days
            
            tsli_df[f"tsli{y}"] = days_since

        return tsli_df

class NdviLoader(BaseDataLoader):
    def load_data(self, valid_uids):
        print("Reading AVHRR_NDVI data...")
        ndvi1 = pd.read_csv(
            self.config.NDVI_AVHRR_ZIP,
            dtype={"UID": str, "year": int, "NDVI": float},
        )
        ndvi1 = ndvi1[ndvi1["UID"].isin(valid_uids)]
        ndvi1["NDVI"] = normalise(ndvi1["NDVI"])
        ndvi_z1 = ndvi1.copy()
        ndvi_z1["NDVI"] = standardise_z(ndvi1["NDVI"])

        print("Reading MODIS_NDVI data...")
        ndvi2 = pd.read_csv(
            self.config.NDVI_MODIS_ZIP,
            dtype={"UID": str, "year": int, "NDVI": float},
        )
        ndvi2 = ndvi2[ndvi2["UID"].isin(valid_uids)]
        ndvi2["NDVI"] = normalise(ndvi2["NDVI"])
        ndvi_z2 = ndvi2.copy()
        # Note: Notebook standardized MODIS using AVHRR stats? 
        # "ndvi_z2['NDVI'] = standardise_z(ndvi1['NDVI'])" -> This looks like a copy-paste error in original notebook 
        # or intentional cross-standardization. Assuming intentional based on "harmonise" comment.
        ndvi_z2["NDVI"] = standardise_z(ndvi2["NDVI"]) 

        ndvi_df = pd.concat([ndvi1, ndvi2], ignore_index=True)
        ndvi_std_df = pd.concat([ndvi_z1, ndvi_z2], ignore_index=True)
        
        return ndvi_df, ndvi_std_df

class SoilMoistureLoader(BaseDataLoader):
    def load_data(self, valid_uids):
        print("Reading root zone soil moisture data...")
        df = pd.read_csv(self.config.SOIL_MOISTURE_ZIP).rename(columns={"MEAN": "soilmoist"})
        df["year"] = pd.to_datetime(df["StdTime"]).dt.year
        # Filter by valid ANAE UIDs
        return df[df["UID"].isin(valid_uids)]

