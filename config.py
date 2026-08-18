"""
Configuration Manager for Basin-scale ANAE Vegetation Vulnerability Analysis.

Provides a unified interface for loading pipeline-specific configurations from YAML files
using Pydantic models for validation and type safety.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class BaseConfig(BaseModel):
    """Base configuration with shared paths and constants."""

    # Core paths (relative to project root)
    data_path: Path = Field(default=Path("data"), description="Base data directory")
    log_path: Path = Field(default=Path("log"), description="Log output directory")
    shapefile_path: Path = Field(
        default=Path("input/spatial/ANAEv3_WIT.shp"), description="ANAE shapefile path"
    )
    poly_unique_id: str = Field(default="UID", description="Unique polygon identifier column")
    debug: bool = Field(default=True, description="Enable debug mode")

    # Computed paths (set during initialization)
    project_root: Optional[Path] = Field(default=None, description="Project root directory")

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Set project root and resolve relative paths
        if self.project_root is None:
            self.project_root = Path(__file__).resolve().parent

        # Convert relative paths to absolute
        self.data_path = self.project_root / self.data_path
        self.log_path = self.project_root / self.log_path
        self.shapefile_path = self.project_root / self.shapefile_path


class NDVILandsatConfig(BaseConfig):
    """Configuration for Landsat NDVI processing via DEA."""

    # STAC and collection settings
    stac_url: str = Field(default="https://explorer.dea.ga.gov.au/stac")
    landsat_collections: List[str] = Field(
        default=["ga_ls5t_ard_3", "ga_ls7e_ard_3", "ga_ls8c_ard_3", "ga_ls9c_ard_3"]
    )
    wofs_collection: str = Field(default="ga_ls_wo_3")

    # Spatial and processing parameters
    crs: str = Field(default="EPSG:3577")
    pixel_size: int = Field(default=30)
    tile_pixels: int = Field(default=2048)
    macro_tile_factor: int = Field(default=4)

    # Temporal parameters
    start_year: int = Field(default=2025)
    start_date: tuple = Field(default=(1988, 1))
    end_date: Optional[tuple] = Field(default=None)

    # Processing parameters
    wofs_filter_mask: int = Field(default=0b01100011)
    max_dask_tasks: int = Field(default=50000)
    max_dask_partitions: int = Field(default=500)

    # Output path (computed)
    output_path: Optional[Path] = Field(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.output_path is None:
            self.output_path = self.data_path / "ndvi/landsat"


class SoilMoistureConfig(BaseConfig):
    """Configuration for Soil Moisture processing via NCI THREDDS."""

    # THREDDS server settings
    thredds_awo_root_zone_soil_moisture_base_url: str = Field(
        default="https://thredds.nci.org.au/thredds/ncss/grid/iu04/australian-water-outlook/historical/v1/AWRALv7/processed/deciles/month/sm_pct.nc"
    )
    local_root_zone_soil_moisture_relative_netcdf_path: Path = Field(
        default=Path("input/spatial/sm_pct.nc")
    )

    # Data processing settings
    sm_var: str = Field(default="sm_pct")
    start_date: str = Field(default="1986-01-31")
    end_date: Optional[str] = Field(default=None)
    crs_fallback: str = Field(default="EPSG:4326")

    # Dask processing parameters
    dask_n_workers_override: int = Field(default=16)
    batch_size: int = Field(default=12)
    block_size: int = Field(default=5000)

    # Computed paths
    output_path: Optional[Path] = Field(default=None)
    root_zone_soil_moisture_netcdf_path: Optional[Path] = Field(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.output_path is None:
            self.output_path = self.data_path / "soil_moisture"
        if self.root_zone_soil_moisture_netcdf_path is None:
            self.root_zone_soil_moisture_netcdf_path = (
                self.project_root / self.local_root_zone_soil_moisture_relative_netcdf_path
            )


class WITMetricsConfig(BaseConfig):
    """Configuration for WIT metrics processing."""

    # Input data settings
    wit_csv_path: Path = Field(default=Path("m:\\ANAE_MDB_WIT_Feb_2026"))
    pc_missing_threshold: float = Field(default=0.1)
    wit_feature_id: str = Field(default="feature_id")

    # Processing options
    interpolate_to_daily: bool = Field(default=True)
    save_interpolated_csv: bool = Field(default=False)
    debug_event_times: bool = Field(default=True)

    # Monthly subset for memory management
    monthly_subset: Optional[List[str]] = Field(
        default=["feature_id", "date", "water+wet_median", "npv+pv+wet_median", "count"]
    )

    # Processing parameters
    batch_size: int = Field(default=500)
    tag: str = Field(default="MDB")
    zip_result: bool = Field(default=True)

    # Inundation thresholds
    threshold_percentile: float = Field(default=0.3)
    min_threshold: float = Field(default=0.05)
    max_threshold: float = Field(default=0.5)

    # Computed paths
    output_path: Optional[Path] = Field(default=None)
    shapefile_key: Optional[str] = Field(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.output_path is None:
            self.output_path = self.data_path / "wit_metrics"
        if self.shapefile_key is None:
            self.shapefile_key = self.poly_unique_id


class VegetationConfig(BaseConfig):
    """Configuration for Vegetation Vulnerability Analysis."""

    # Input data paths
    csv_input_path: Path = Field(default=Path("data/wit-metrics"))
    ndvi_csv_input_path: Path = Field(default=Path("data/ndvi"))
    soil_moisture_csv_input_path: Path = Field(default=Path("data/soil_moisture"))

    # Analysis parameters
    millennium_drought: List[int] = Field(default=list(range(2001, 2011)))
    veg_window_width: int = Field(default=5)
    veg_trend_width: int = Field(default=2)

    # Spatial data files
    valley_shp: Path = Field(default=Path("input/spatial/BWSRegions.shp"))
    diwa_shp: Path = Field(default=Path("input/spatial/DIWA_complex.shp"))
    ramsar_shp: Path = Field(default=Path("input/spatial/ramsar_wetlands.shp"))

    # Input data files
    wit_metrics_zip: Path = Field(default=Path("data/wit_metrics/MDB_WIT_monthly_metrics.zip"))
    wit_tsli_zip: Path = Field(
        default=Path("data/wit_metrics/MDB_WIT_time_since_last_inundation.zip")
    )
    wit_inundation_zip: Path = Field(
        default=Path("data/wit_metrics/MDB_WIT_inundation_metrics.zip")
    )
    ndvi_path: Path = Field(default=Path("data/ndvi/landsat"))
    soil_moisture_zip: Path = Field(
        default=Path("data/soilmoisture/ZonalSt30_soilmoistureanomally.zip")
    )

    # TSLI stress thresholds by vegetation type
    vegetation_tsli_stress_thresholds: Dict[str, List[int]] = Field(
        default={
            "river red gum swamps and forest": [0, 730, 1825],
            "river red gum woodland": [0, 1460, 2555],
            "black box swamps and forest": [0, 1095, 2555],
            "black box woodland": [0, 1825, 3285],
            "cooba woodland": [0, 1460, 2555],
            "coolibah swamp": [0, 2555, 5475],
            "coolibah woodland": [0, 3650, 7300],
            "lignum swamp": [0, 1095, 1825],
            "lignum shrubland": [0, 1825, 2555],
            "shrubland": [0, 1460, 3650],
            "woodland and shrubland swamp": [0, 1095, 2555],
            "woodland (other)": [0, 1825, 3285],
            "permanent lakes": [0, 182, 365],
            "temporary lakes": [0, 1095, 1825],
            "tall emergent marsh": [0, 365, 730],
            "permanent aquatic meadow": [0, 365, 547],
            "temporary herbfield": [0, 365, 1460],
            "clay pan": [0, 3650, 7300],
        }
    )

    # ANAE type to functional group mappings (loaded from YAML)
    anae_type_to_functional_group: Dict[str, List[str]] = Field(default={})

    # ANAE groupings (computed during initialization)
    anae_groupings: Optional[Dict[str, str]] = Field(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Convert relative paths to absolute
        self.valley_shp = self.project_root / self.valley_shp
        self.diwa_shp = self.project_root / self.diwa_shp
        self.ramsar_shp = self.project_root / self.ramsar_shp
        self.wit_metrics_zip = self.project_root / self.wit_metrics_zip
        self.wit_tsli_zip = self.project_root / self.wit_tsli_zip
        self.wit_inundation_zip = self.project_root / self.wit_inundation_zip
        self.ndvi_path = self.project_root / self.ndvi_path
        self.soil_moisture_zip = self.project_root / self.soil_moisture_zip

        # Build ANAE groupings mapping from YAML data
        if self.anae_groupings is None:
            self.anae_groupings = self._build_anae_groupings()

    def _build_anae_groupings(self) -> Dict[str, str]:
        """Build the reverse mapping from ANAE types to functional groups using YAML data."""
        reverse_mapping = {}

        # Use the YAML-loaded mappings
        for group, anae_types in self.anae_type_to_functional_group.items():
            for anae_type in anae_types:
                reverse_mapping[anae_type] = group

        return reverse_mapping


# Mapping of config names to their Pydantic models
CONFIG_MODELS = {
    "ndvi_landsat": NDVILandsatConfig,
    "soil_moisture": SoilMoistureConfig,
    "wit_metrics": WITMetricsConfig,
    "vegetation": VegetationConfig,
}


def load_config(name: str, config_file: Optional[Path] = None) -> BaseConfig:
    """
    Load a configuration by name from YAML file.

    Args:
        name: Configuration name (e.g., 'vegetation', 'ndvi_landsat')
        config_file: Optional path to specific config file. If None, uses default naming.

    Returns:
        Loaded and validated configuration object

    Raises:
        ValueError: If config name is not recognized
        FileNotFoundError: If config file doesn't exist
        RuntimeError: If config loading or validation fails
    """
    if name not in CONFIG_MODELS:
        valid_configs = ", ".join(CONFIG_MODELS.keys())
        raise ValueError(f"Unknown config: {name}. Valid configs are: {valid_configs}")

    # Determine config file path
    if config_file is None:
        config_dir = Path(__file__).parent / "configs"
        config_file = config_dir / f"{name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load YAML data
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_file}: {e}") from e

    # Create and validate config object
    config_class = CONFIG_MODELS[name]
    try:
        config = config_class(**yaml_data)
        logger.info("Loaded %s configuration from %s", name, config_file)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to validate {name} configuration: {e}") from e


# Maintain backward compatibility
CONFIGS = CONFIG_MODELS
