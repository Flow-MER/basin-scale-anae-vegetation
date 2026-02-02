"""
Configuration settings for FlowMER2.0 vegetation vulnerability processing.

Defines paths, constants, and parameters for:
- Landsat NDVI processing (DEA)
- AVHRR/MODIS NDVI harmonization (GEE)
- Soil Moisture processing (AWO/THREDDS)
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
import os
import multiprocessing as mp

@dataclass(frozen=True)
class BaseConfig:
    """Base configuration with shared paths and constants."""
    BASE_DIR: Path = Path(__file__).parent
    INPUT_DIR: Path = BASE_DIR / "input/spatial"
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "log"
    TOOLS_DIR: Path = BASE_DIR / "tools"
    POLYGON_PATH: Path = INPUT_DIR / "ANAEv3_WIT.shp"
    POLY_UNIQUE_ID: str = "UID"
    USE_CACHE: bool = True


@dataclass(frozen=True)
class NDVILandsatConfig(BaseConfig):
    """Configuration for Landsat NDVI processing via DEA."""
    TASK_NAME: str = "ndvi"
    OUTPUT_DIR: Path = BaseConfig.DATA_DIR / "ndvi/landsat"
    STAC_URL: str = "https://explorer.dea.ga.gov.au/stac"
    LANDSAT_COLLECTIONS: List[str] = field(default_factory=lambda: [
        "ga_ls5t_ard_3", "ga_ls7e_ard_3", "ga_ls8c_ard_3", "ga_ls9c_ard_3"])
    WOFS_COLLECTION: str = "ga_ls_wo_3"
    CRS: str = "EPSG:3577"
    PIXEL_SIZE: int = 30
    TILE_PIXELS: int = 2048
    MACRO_TILE_FACTOR: int = 4
    START_YEAR: int = 2025
    START_DATE: tuple = (1988, 1)
    END_DATE: tuple | None = None
    WOFS_FILTER_MASK: int = 0b01100011
    DASK_DASHBOARD_PORT: int = 8787
    MAX_DASK_TASKS: int = 50000
    MAX_DASK_PARTITIONS: int = 500

@dataclass(frozen=True)
class SoilMoistureConfig(BaseConfig):
    """Configuration for Soil Moisture processing via NCI THREDDS."""
    OUTPUT_DIR: Path = BaseConfig.DATA_DIR / "soil_moisture"
    CACHE_DIR: Path = OUTPUT_DIR / "cache"
    THREDDS_AWO_ROOT_ZONE_SOIL_MOISTURE_BASE_URL: str = (
        "https://thredds.nci.org.au/thredds/ncss/grid/iu04/australian-water-outlook/historical/v1/AWRALv7/processed/deciles/month/sm_pct.nc"
    )
    LOCAL_ROOT_ZONE_SOIL_MOISTURE_RELATIVE_NETCDF_PATH: Path = (
        BaseConfig.INPUT_DIR / "sm_pct.nc"
    )
    SM_VAR: str = "sm_pct"
    START_DATE: str = "1986-01-31"
    END_DATE: str | None = None  # None = use last month in netcdf
    CRS_FALLBACK: str = "EPSG:4326" #WGS_84 matches netcdf
    DASK_N_WORKERS_OVERRIDE: int = (
        16  # Number of Dask workers to use this script benefits from more workers
    )
    BATCH_SIZE: int = 12  # Number of months to process in one Dask batch
    BLOCK_SIZE: int = 5000  # Number of polygons per Dask worker to process together.


@dataclass(frozen=True)
class WITMetricsConfig(BaseConfig):
    """Configuration for WIT Metrics processing pipeline."""
    OUTPUT_DIR: Path = BaseConfig.DATA_DIR / "wit_metrics"

    # Path to folder that contains the WIT csv files to process.  When debugging providing a single file might be prudent.
    WIT_CSV_PATH = Path("M:/WIT_TEST/")

# shapefile: the shape file mentioned above to find the  and get their area
    # set to '' to disable area lookup
    SHAPEFILE_PATH: Path = BaseConfig.POLYGON_PATH

    # shapefile field name that identifies each polygon -  the ANAEv3 UID geohash was used here.
    # The ANAE UID is also used in the naming convention for the CSV files
    SHAPEFILE_KEY: str = BaseConfig.POLY_UNIQUE_ID

    # Only use WIT data where the pc_missing is less than the threshold (default is 0.1) i.e. >90% of the polygon was visible to satellites
    PC_MISSING_THRESHOLD: float = 0.1

    # csv feature_id - the WIT csv output files include a column 'feature_id' that in this case is the ANAE UID
    WIT_FEATURE_ID: str = "feature_id"

    # whether to interpolate the WIT observation dates (typically 10-50+ per year) to daily data (365 per year)
    # This is computationally expensive but improves estimates of inundation duration and time since last inundation.
    # Monthly WIT stats require interpolated daily data to infill missing records and will not be generated if interpolate_to_daily = False
    INTERPOLATE_TO_DAILY: bool = True

    # set to True to save the interpolated daily WIT csv in a subfolder under the csv_files (unnecessary and take up a lot of space but good for debugging)
    SAVE_INTERPOLATED_CSV: bool = False

    # monthly metrics files are too big for most computers when all metrics are used (e.g. just 4 metrics x 270,000 polygons x 450 months is a 5GB csv file)
    # specify a subset that will be joined together into the monthly result - must include ["feature_id","date", at-least-one-metric]\
    MONTHLY_SUBSET: list | None = None  # do not prune
    # monthly_subset=[
    #         "feature_id",
    #         "date",
    #         "water_median",
    #         "wet_median",
    #         "pv_median",
    #         "npv_median",
    #         "bs_median",
    #         "count",
    #     ]

    # set to true to save intermediate data frames containing the event times and stats
    # these are saved in the working directory
    DEBUG_EVENT_TIMES: bool = True

    # batchsize is the number of WIT csv files to include in each 'batch' that is processed by each single CPU core.
    # The code was designed to process several 100,000 polygons in small batches that fit into the computer memory
    # then glue all the batch results together at the end.
    # On a workstation with 16 cpu cores and 64MB RAM a batchsize of 100-200 worked well. With smaller number of CSV a batch size < total number of CSV allows the
    # calculations to be spread across multiple processors.
    # during processing the code will generate outputs for each batch then glue them together at the end.

    BATCH_SIZE: int = 100

    # tag prepended to final result files (zipped csv)
    TAG: str = "RESULT"

    # Whether to zip the final result csv to save space (python/pandas can read the csv from the zips)
    ZIP_RESULT: bool = False

    
    # define threshold and bounds to define inundation events
    THRESHOLD_PERCENTILE: float = (
        0.3  # 0.3 = 30th percentile of 'water+wet' area (lower than 0.5 keeps the median in the inundated state)
    )
    MIN_THRESHOLD: float = (
        0.05  # Floor for dry sites. (0.05=5% area) stops noise and minimal water detection inclusion as inundation event
    )
    MAX_THRESHOLD: float = (
        0.50  # Cap for very wet sites. permanent lake is still considered inundated until falls below 50% water by area
    )


@dataclass(frozen=True)
class VegetationConfig(BaseConfig):
    """Configuration for Vegetation Vulnerability Analysis."""

    CSV_INPUT_DIR: Path = BaseConfig.BASE_DIR / "input/csv"

    # Year ranges
    ALL_TIME: List[int] = field(default_factory=lambda: list(range(1987, 2025)))
    MILLENNIUM_DROUGHT: List[int] = field(
        default_factory=lambda: list(range(2001, 2010))
    )

    # Analysis parameters
    VEG_WINDOW_WIDTH: int = 5
    VEG_TREND_WIDTH: int = 2

    # Thresholds
    VEGETATION_TSLI_STRESS_THRESHOLDS: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "river red gum swamps and forests": [0, 730, 1825],
            "river red gum woodland": [0, 730, 1825],
            "black box": [0, 1460, 2555],
            "cooba": [0, 1460, 2555],
            "coolibah": [0, 3650, 7300],
            "lignum": [0, 1095, 2555],
            "shrubland": [0, 1095, 3650],
            "submerged lake": [0, 90, 120],
            "tall reed beds": [0, 365, 730],
            "grassy meadows": [0, 240, 300],
            "herbfield": [0, 365, 1460],
        }
    )

    # Input Files
    ANAE_SHP: Path = BaseConfig.INPUT_DIR / "ANAEv3_BWS.shp"
    VALLEY_SHP: Path = BaseConfig.INPUT_DIR / "BWSRegions.shp"
    DIWA_SHP: Path = BaseConfig.INPUT_DIR / "DIWA_complex.shp"
    RAMSAR_SHP: Path = BaseConfig.INPUT_DIR / "ramsar_wetlands.shp"

    WIT_METRICS_ZIP: Path = CSV_INPUT_DIR / "RESULT_WIT_ANAE_yearly_metrics.zip"
    WIT_TSLI_ZIP: Path = (
        CSV_INPUT_DIR / "RESULT_WIT_ANAE_time_since_last_inundation.zip"
    )
    WIT_INUNDATION_ZIP: Path = CSV_INPUT_DIR / "RESULT_WIT_ANAE_inundation_metrics.zip"
    NDVI_AVHRR_ZIP: Path = CSV_INPUT_DIR / "NDVI_1986-2000_ANAEv3_annual_AVHRR.zip"
    NDVI_MODIS_ZIP: Path = CSV_INPUT_DIR / "NDVI_2001-2024_ANAEv3_annual_MODIS.zip"
    SOIL_MOISTURE_ZIP: Path = CSV_INPUT_DIR / "ZonalSt30_soilmoistureanomally.zip"


ndvi_landsat_cfg = NDVILandsatConfig()
soil_moisture_cfg = SoilMoistureConfig()
wit_metrics_cfg = WITMetricsConfig()
veg_config = VegetationConfig()
veg_config = VegetationConfig()
