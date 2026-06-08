# Configuration System

This directory contains YAML configuration files for the Basin-scale ANAE Vegetation Vulnerability analysis pipelines.

## Configuration Files

- **`ndvi_landsat.yaml`** - Configuration for Landsat NDVI processing via DEA STAC
- **`soil_moisture.yaml`** - Configuration for soil moisture processing via NCI THREDDS  
- **`wit_metrics.yaml`** - Configuration for WIT metrics processing
- **`vegetation.yaml`** - Main configuration for vegetation vulnerability analysis

## Usage

### Loading Configurations

```python
from config_manager import load_config

# Load a specific configuration
config = load_config("vegetation")
config = load_config("ndvi_landsat")
config = load_config("soil_moisture")
config = load_config("wit_metrics")
```

### Custom Configuration Files

```python
from pathlib import Path
config = load_config("vegetation", Path("my_custom_config.yaml"))
```

### Configuration Structure

All configurations inherit from a base configuration with common settings:

```yaml
# Base configuration (inherited by all)
data_path: "data"
log_path: "log"
shapefile_path: "input/spatial/ANAEv3_WIT.shp"
poly_unique_id: "UID"
debug: true
```

### Path Resolution

- All paths in YAML files are relative to the project root
- Paths are automatically resolved to absolute paths when loaded
- Computed paths (like `output_path`) are generated based on the configuration

### Validation

Configurations are validated using Pydantic models:
- Type checking for all fields
- Required field validation
- Custom validation rules for specific fields
- Helpful error messages for invalid configurations

## Migration from Old System

The old `config.py` system is still supported for backward compatibility but will show deprecation warnings. To migrate:

**Old way:**
```python
from config import load_config, VegetationConfig
```

**New way:**
```python
from config_manager import load_config, VegetationConfig
```

## Customization

### Environment-Specific Configurations

Create environment-specific config files:

```
configs/
├── vegetation.yaml          # Default
├── vegetation_dev.yaml      # Development
├── vegetation_prod.yaml     # Production
└── vegetation_test.yaml     # Testing
```

Load with:
```python
config = load_config("vegetation", Path("configs/vegetation_dev.yaml"))
```

## Configuration Reference

### NDVI Landsat (`ndvi_landsat.yaml`)

Key settings:
- `stac_url`: DEA STAC catalog URL
- `landsat_collections`: List of Landsat collection IDs
- `pixel_size`: Processing resolution in meters
- `start_date`/`end_date`: Temporal processing range

### Soil Moisture (`soil_moisture.yaml`)

Key settings:
- `thredds_awo_root_zone_soil_moisture_base_url`: NCI THREDDS server URL
- `dask_n_workers_override`: Number of Dask workers
- `batch_size`: Months processed per batch

### WIT Metrics (`wit_metrics.yaml`)

Key settings:
- `wit_csv_path`: Directory containing WIT CSV files
- `interpolate_to_daily`: Enable daily interpolation
- `threshold_percentile`: Inundation detection threshold

### Vegetation (`vegetation.yaml`)

Key settings:
- `millennium_drought`: Years to exclude from baseline
- `vegetation_tsli_stress_thresholds`: TSLI thresholds by vegetation type
- `veg_window_width`/`veg_trend_width`: Analysis window sizes