"""
Data loaders for Vegetation Vulnerability Analysis.
Handles loading of Spatial, WIT, NDVI, and Soil Moisture datasets with memory optimizations.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from config import VegetationConfig

logger = logging.getLogger(__name__)


class SpatialDataManager:
    """
    Manages spatial data loading and aggregation hierarchies with memory optimizations.
    Handles ANAE polygons and spatial joins for different aggregation levels.
    """

    def __init__(self, config: VegetationConfig, anae_cache: Path = None):
        self.config = config

        # Try to load from cache if provided
        if anae_cache and anae_cache.exists():
            try:
                self._load_cached_anae(anae_cache)
            except Exception as e:
                logger.warning(f"Failed to load cache {anae_cache}: {e}")
                self.anae_gdf = None
                self.aggregators = {}
                self.agg_fields = {}
        else:
            self.anae_gdf = None
            self.aggregators = {}
            self.agg_fields = {}

    def _load_cached_anae(self, anae_cache: Path):
        """
        Loads cached ANAE data with optimized data types.

        Args:
            anae_cache (Path): Path to cached ANAE parquet file
        """
        logger.info(f"Loading cached ANAE data from {anae_cache}")
        self.anae_gdf = gpd.read_parquet(anae_cache)

        # Optimize data types for memory efficiency
        self.anae_gdf["UID"] = self.anae_gdf["UID"].astype("category")
        self.uid_dtype = self.anae_gdf["UID"].dtype
        if "grp" in self.anae_gdf.columns:
            self.anae_gdf["grp"] = self.anae_gdf["grp"].astype("category")

        # Set up basic aggregation structure
        self.aggregators = {"Valley"}
        self.agg_fields = {"Valley": "BWS_Region"}

        # Validate required columns
        all_agg_cols = []
        for val in self.agg_fields.values():
            if isinstance(val, (list, tuple)):
                all_agg_cols.extend(val)
            else:
                all_agg_cols.append(val)

        required_cols = ["UID", "Area_Ha", "grp"] + all_agg_cols
        self._validate_columns(self.anae_gdf, required_cols, anae_cache)

    def _assign_group_name(self, anae_type):
        """
        Maps ANAE types to functional groups using configuration.

        Args:
            anae_type (str): ANAE vegetation type

        Returns:
            str: Functional group name or None if not mapped
        """
        anae_type = str(anae_type).lower()
        return self.config.anae_groupings.get(anae_type)

    def _validate_columns(self, gdf, columns, gdf_path):
        """
        Validates that required columns exist in the GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to validate
            columns (list): Required column names
            gdf_path (Path): Path for error reporting

        Raises:
            Exception: If required columns are missing
        """
        missing_cols = set(columns) - set(gdf.columns)
        if missing_cols:
            raise Exception(f"Missing columns {missing_cols} in {gdf_path}. Required: {columns}")

    def load_anae(self):
        """
        Loads ANAE shapefile and sets up aggregation hierarchies.
        Optimizes memory usage through selective column loading and categorical data types.
        """
        logger.info("Loading ANAE shapefile...")

        if not self.config.shapefile_path or not Path(self.config.shapefile_path).exists():
            raise FileNotFoundError(f"ANAE shapefile not found: {self.config.shapefile_path}")

        # Load only required columns to reduce memory usage
        self.anae_gdf = gpd.read_file(self.config.shapefile_path).to_crs("EPSG:3577")
        self._validate_columns(
            self.anae_gdf, ["UID", "ANAE_TYPE", "Area_Ha", "geometry"], self.config.shapefile_path
        )

        # Select and optimize columns
        self.anae_gdf = self.anae_gdf[["UID", "ANAE_TYPE", "Area_Ha", "geometry"]].copy()
        self.anae_gdf["UID"] = self.anae_gdf["UID"].astype("category")
        # preserve index to assign to other loaded data sets (WIT, soil moisture etc for efficiency of same index)
        self.uid_dtype = self.anae_gdf["UID"].dtype

        # Apply functional grouping
        self.anae_gdf["grp"] = self.anae_gdf["ANAE_TYPE"].apply(self._assign_group_name)
        self.anae_gdf["grp"] = self.anae_gdf["grp"].astype("category")
        # Drop ANAE_TYPE (strings) to save memory
        self.anae_gdf = self.anae_gdf.drop(columns="ANAE_TYPE")

        # Set up aggregation hierarchies
        self._setup_aggregators()

    def _setup_aggregators(self):
        """
        Sets up spatial aggregation hierarchies with memory-efficient indexing.
        """
        self.aggregators = {}
        self.agg_fields = {}

        # Basin scale aggregation (no spatial grouping)
        basin_data = self.anae_gdf[["UID", "grp", "Area_Ha"]].dropna(subset=["grp"])
        self.aggregators["Basin"] = basin_data.set_index("UID")
        self.aggregators["Basin"].name = "Basin"
        self.agg_fields["Basin"] = []

        # ANAE scale (individual polygons) - drop geometry to save memory
        anae_data = self.anae_gdf.drop(columns=["geometry"])
        self.aggregators["ANAE"] = anae_data.set_index("UID")
        self.aggregators["ANAE"].name = "ANAE"
        self.agg_fields["ANAE"] = ["UID"]

    def load_data(self, spatial_aggregator_shape_file, name=None, unique_id=None):
        """
        Loads additional spatial aggregation layers with memory-optimized spatial joins.

        Args:
            spatial_aggregator_shape_file (Path): Path to aggregation shapefile
            name (str): Name for the aggregation level
            unique_id (str or list): Unique identifier column(s)

        Returns:
            gpd.GeoDataFrame: Spatially joined data

        Raises:
            RuntimeError: If name or unique_id not specified
            FileNotFoundError: If shapefile doesn't exist
        """
        if name is None or unique_id is None:
            raise RuntimeError("Must specify name and unique_id for aggregator")

        if isinstance(unique_id, str):
            unique_id = [unique_id]

        try:
            if not spatial_aggregator_shape_file.exists():
                raise FileNotFoundError(f"Shapefile not found: {spatial_aggregator_shape_file}")

            logger.info(f"Loading spatial aggregator: {spatial_aggregator_shape_file.name}")
            self.agg_fields[name] = unique_id

            # Load only required columns to minimize memory usage
            required_cols = unique_id + ["geometry"]
            gdf = gpd.read_file(self.config.valley_shp)[required_cols].to_crs("EPSG:3577")

            # Memory-optimized spatial join
            gdf_sj = (
                gpd.sjoin(self.anae_gdf, gdf, how="left", predicate="intersects")
                .dropna(subset=["grp"])
                .drop(columns=["index_right"])
                .set_index("UID")
            )

            # Optimize categorical columns
            for col in unique_id:
                if col in gdf_sj.columns:
                    gdf_sj[col] = gdf_sj[col].astype("category")

            if not gdf_sj.empty:
                self.aggregators[name] = gdf_sj
                self.aggregators[name].name = name
                # Update main ANAE GDF with spatial join results
                self.anae_gdf = gdf_sj

            return gdf_sj

        except Exception as e:
            logger.error(f"Error loading spatial aggregator {name}: {e}")
            raise


class BaseDataLoader:
    """Base class for data loaders with common functionality."""

    def __init__(self, config: VegetationConfig):
        self.config = config


class WitMetricsLoader(BaseDataLoader):
    """
    Loads WIT (Wetland Insights Tool) metrics with memory optimizations.
    """

    def load_data(self, uid_dtype):
        """
        Loads WIT metrics data filtered by valid UIDs.

        Args:
            uid_dtype: Master categorical index of UID from shapefile.

        Returns:
            pd.DataFrame: WIT metrics with date column as datetime

        Raises:
            ValueError: If no data found for specified UIDs
            KeyError: If required columns are missing
        """
        logger.info(f"Loading WIT metrics from {self.config.wit_metrics_zip}")

        try:
            # Load data with optimized data types any UID not in uid_dtype will become NaN and dropped
            wit_metrics_df = pd.read_csv(
                self.config.wit_metrics_zip,
                dtype={"feature_id": uid_dtype},  # Optimize UID column
            ).rename(columns={"feature_id": "UID"})

        except Exception as e:
            logger.error(f"Error reading WIT metrics: {e}")
            raise

        # Validate required columns
        if "date" not in wit_metrics_df.columns:
            raise KeyError("Column 'date' not found in WIT metrics")

        # Drop rows with invalid UID (not in master categories defined in uid_dtype)
        wit_metrics_df = wit_metrics_df[wit_metrics_df["UID"].notna()]

        if wit_metrics_df.empty:
            raise ValueError("No WIT data found for the specified UIDs")

        # Convert date column efficiently and convert to month-end dates for TSLI joining
        wit_metrics_df["date"] = pd.to_datetime(wit_metrics_df["date"]) + pd.offsets.MonthEnd(0)

        return wit_metrics_df


class TsliLoader(BaseDataLoader):
    """
    Loads Time Since Last Inundation (TSLI) data with memory optimizations.
    """

    def load_data(self, uid_dtype):
        """
        Loads inundation event data for TSLI calculation.

        Args:
            uid_dtype: Master categorical index of UID from shapefile.

        Returns:
            pd.DataFrame: Inundation events with duration and gap in days
        """
        logger.info(f"Loading inundation metrics from {self.config.wit_inundation_zip}")
        try:
            # Load with optimized parsing
            inundation_df = pd.read_csv(
                self.config.wit_inundation_zip,
                parse_dates=["start_date", "end_date"],
                dtype={"feature_id": uid_dtype},
            ).rename(columns={"feature_id": "UID"})

        except Exception as e:
            logger.error(f"Error reading inundation metrics: {e}")
            raise

        if inundation_df.empty:
            raise ValueError("No TSLI data found")

        # Drop rows with invalid UID (not in master categories defined in uid_dtype)
        inundation_df = inundation_df[inundation_df["UID"].notna()]

        # Convert duration and gap to days (more memory efficient than timedelta)
        inundation_df["duration"] = pd.to_timedelta(inundation_df["duration"]).dt.days
        inundation_df["gap"] = pd.to_timedelta(inundation_df["gap"]).dt.days

        return inundation_df


class NdviLoader(BaseDataLoader):
    """
    Loads NDVI data with memory-optimized interpolation for missing values.
    """

    def _complete_time_index(df):
        # Create full date range
        full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="M")

        # Build full index
        full_index = pd.MultiIndex.from_product(
            [df["UID"].cat.categories, full_dates], names=["UID", "date"]
        )

        # Reindex to full grid
        df = df.set_index(["UID", "date"]).reindex(full_index).reset_index()

        return df

    def load_data(self, uid_dtype, needed_cols=None):
        """
        Loads and interpolates NDVI data with memory-efficient processing.
        Uses chunked processing to avoid loading all files simultaneously.

        Args:
            uid_dtype: Master categorical index of UID from shapefile.

        Returns:
            pd.DataFrame: NDVI data with interpolated missing values
        """

        logger.info(f"Loading NDVI data from {self.config.ndvi_path}")
        if needed_cols is None:
            needed_cols = needed_cols or ["UID", "year", "month", "ndvi"]

        files = list(self.config.ndvi_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No NDVI parquet files found in {self.config.ndvi_path}")
        # optimise loading to just the needed columns to reduce memory usage

        parquet_file = pq.ParquetFile(files[0])
        parquet_cols = parquet_file.schema.names
        if needed_cols is None:
            needed_cols = parquet_cols
        else:
            # raise exception if any needed_cols not in parquet_cols
            missing_cols = set(needed_cols) - set(parquet_cols)
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in parquet file")

        # Load all parquet files
        ndvi_df = pd.concat(
            (pd.read_parquet(f, columns=needed_cols) for f in files), ignore_index=True
        )

        if ndvi_df.empty:
            raise ValueError("No NDVI data found")

        # Apply master UID dtype immediately
        ndvi_df["UID"] = ndvi_df["UID"].astype(uid_dtype)

        # Drop invalid UIDs (not in category set)
        ndvi_df = ndvi_df[ndvi_df["UID"].notna()]

        # Create month-end dates
        ndvi_df["date"] = pd.to_datetime(
            ndvi_df[["year", "month"]].assign(day=1)
        ) + pd.offsets.MonthEnd(0)

        # Keep only required columns
        ndvi_df = ndvi_df[["UID", "date", "ndvi"]]

        # Sort for correct interpolation
        ndvi_df = ndvi_df.sort_values(["UID", "date"])

        # Vectorized groupby-transform interpolation of any missing values
        ndvi_df["ndvi"] = ndvi_df.groupby("UID")["ndvi"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

        return ndvi_df


class SoilMoistureLoader(BaseDataLoader):
    """
    Loads soil moisture data with memory optimizations.
    """

    def load_data(self, uid_dtype, needed_cols=None):
        """
        Loads soil moisture data filtered by valid UIDs.

        Args:
            uid_dtype: Master categorical index of UID from shapefile.

        Returns:
            pd.DataFrame: Soil moisture data with date column
        """
        logger.info("Loading soil moisture data from cache")
        if needed_cols is None:
            needed_cols = ["UID", "year", "month", "sm_pct"]

        if uid_dtype is None:
            raise ValueError("uid_dtype must be provided")

        # Load cached soil moisture data
        cache_path = self.config.data_path / "soil_moisture/cache"
        if not cache_path.exists():
            raise FileNotFoundError(f"Soil moisture cache not found: {cache_path}")

        files = list(cache_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No soil moisture parquet files found in {cache_path}")

        parquet_file = pq.ParquetFile(files[0])
        parquet_cols = parquet_file.schema.names
        if needed_cols is None:
            needed_cols = parquet_cols
        else:
            # raise exception if any needed_cols not in parquet_cols
            missing_cols = set(needed_cols) - set(parquet_cols)
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in parquet file")

        # Load all parquet files
        soil_moisture_df = pd.concat(
            (pd.read_parquet(f, columns=needed_cols) for f in files), ignore_index=True
        )

        if soil_moisture_df.empty:
            raise ValueError("No soil moisture data found")

        # Apply master UID dtype immediately
        soil_moisture_df["UID"] = soil_moisture_df["UID"].astype(uid_dtype)

        # Drop invalid UIDs (not in category set)
        soil_moisture_df = soil_moisture_df[soil_moisture_df["UID"].notna()]

        # Create month-end dates
        soil_moisture_df["date"] = pd.to_datetime(
            soil_moisture_df[["year", "month"]].assign(day=1)
        ) + pd.offsets.MonthEnd(0)

        # Rename soil moisture column and keep only required columns
        soil_moisture_df = soil_moisture_df.rename(columns={"sm_pct": "soilmoist"})
        soil_moisture_df = soil_moisture_df[["UID", "date", "soilmoist"]]

        return soil_moisture_df
