"""
Configuration settings for FlowMER2.0 vegetation vulnerability processing.

Defines paths, constants, and parameters for:
- Landsat NDVI processing (DEA)
- AVHRR/MODIS NDVI harmonization (GEE)
- Soil Moisture processing (AWO/THREDDS)
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import os
import multiprocessing as mp

@dataclass(frozen=True)
class BaseConfig:
    """Base configuration with shared paths and constants."""
    BASE_DIR: Path = Path(__file__).parent
    INPUT_DIR: Path = BASE_DIR / "input/spatial"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    LOG_DIR: Path = BASE_DIR / "log"
    TOOLS_DIR: Path = BASE_DIR / "tools"
    POLYGON_PATH: Path = INPUT_DIR / "ANAEv3_WIT.shp"
    POLY_UID: str = "UID"

@dataclass(frozen=True)
class NDVILandsatConfig(BaseConfig):
    """Configuration for Landsat NDVI processing via DEA."""
    TASK_NAME: str = "ndvi"
    OUTPUT_DIR: Path = BaseConfig.OUTPUT_DIR / "ndvi/landsat"
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
class NDVIAvhrrModisConfig(BaseConfig):
    """Configuration for AVHRR/MODIS NDVI harmonization via GEE."""
    OUTPUT_DIR: Path = BaseConfig.OUTPUT_DIR / "ndvi/avhrr_modis"
    GEE_ASSET_ID: str = "projects/ee-litepc/assets/ANAEv3_gt1ha"
    START_DATE: tuple = (1986, 1)
    END_DATE: tuple | None = None
    AVHRR_END: tuple = (2013, 12)
    MODIS_START: tuple = (2000, 3)
    OVERLAP_PERIOD: tuple = (2000, 3, 2013, 12)
    SCALE_FACTOR: float = 0.0001
    REDUCE_SCALE: int = 250
    MIN_HARMONIZATION_SAMPLES: int = 24
    MAX_PROCESSOR_COUNT: int = int(os.getenv('MAX_PROCESSOR_COUNT', mp.cpu_count()))
    USE_CLOUD_STORAGE: bool = False
    GCS_BUCKET: str = os.getenv('GCS_BUCKET', 'your-bucket')
    GEE_PROJECT: str = os.getenv('GEE_PROJECT', None)
    SENSOR_COLLECTIONS: dict = field(default_factory=lambda: {
        'AVHRR': 'NOAA/CDR/AVHRR/NDVI/V5',
        'MODIS': 'MODIS/061/MOD13Q1'})
    TEST_POLYGON_LIMIT: int = None
    USE_RANDOM_SAMPLE: bool = False

@dataclass(frozen=True)
class SoilMoistureConfig(BaseConfig):
    """Configuration for Soil Moisture processing via NCI THREDDS."""
    OUTPUT_DIR: Path = BaseConfig.OUTPUT_DIR / "soil_moisture"
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


ndvi_landsat_cfg = NDVILandsatConfig()
ndvi_avhrr_modis_cfg = NDVIAvhrrModisConfig()
soil_moisture_cfg = SoilMoistureConfig()
