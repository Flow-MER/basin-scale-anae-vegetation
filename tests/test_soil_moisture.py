import sys
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
import numpy as np
from pathlib import Path
import dask

from data_pipelines.soil_moisture_monthly_awo import (
    get_parquet_filename,
    get_existing_months,
    get_ncss_time_range,
    download_mdb_soilmoisture_subset,
    process_polygon_block_lazy,
    write_month_parquet,
    aggregate_monthly_results,
)

@pytest.fixture
def mock_config(tmp_path):
    """Mock configuration object with temporary paths."""
    mock_cfg = MagicMock()
    mock_cfg.output_path = tmp_path / "output"
    mock_cfg.shapefile_path = tmp_path / "polygons.shp"
    mock_cfg.local_root_zone_soil_moisture_relative_netcdf_path = tmp_path / "sm.nc"
    mock_cfg.thredds_awo_root_zone_soil_moisture_base_url = "http://mock.url"
    mock_cfg.poly_unique_id = "UID"
    mock_cfg.sm_var = "sm_pct"
    mock_cfg.crs_fallback = "EPSG:4326"
    mock_cfg.start_date = "2020-01-01"
    mock_cfg.end_date = "2020-12-31"
    mock_cfg.block_size = 100
    mock_cfg.batch_size = 10
    mock_cfg.dask_n_workers_override = 1
    return mock_cfg

@pytest.fixture
def sample_gdf():
    """Create a simple GeoDataFrame for testing."""
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    return gpd.GeoDataFrame(
        {"UID": ["A", "B"], "geometry": [p1, p2]},
        crs="EPSG:4326"
    )

@pytest.fixture
def sample_raster():
    """Create a simple xarray DataArray with CRS."""
    data = np.random.rand(1, 10, 10) # Time, Y, X
    coords = {
        "time": [pd.Timestamp("2020-01-01")],
        "y": np.linspace(0, 2, 10),
        "x": np.linspace(0, 2, 10),
    }
    da = xr.DataArray(data, coords=coords, dims=("time", "y", "x"), name="sm_pct")
    da.rio.write_crs("EPSG:4326", inplace=True)
    return da

def test_get_parquet_filename(tmp_path):
    out_path = tmp_path / "cache"
    fname = get_parquet_filename(out_path, 2020, 1)
    assert fname == out_path / "soil_moisture_2020_01.parquet"

def test_get_existing_months(tmp_path):
    (tmp_path / "soil_moisture_2020_01.parquet").touch()
    (tmp_path / "soil_moisture_2020_02.parquet").touch()
    (tmp_path / "other.txt").touch()
    
    existing = get_existing_months(tmp_path)
    assert existing == {(2020, 1), (2020, 2)}

@patch("requests.get")
def test_get_ncss_time_range(mock_get):
    # Success case
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"<dataset><TimeSpan><end>2023-12-31T00:00:00Z</end></TimeSpan></dataset>"
    assert get_ncss_time_range("http://url") == "2023-12-31T00:00:00Z"
    
    # Failure case
    mock_get.side_effect = Exception("Fail")
    assert get_ncss_time_range("http://url") is None

@patch("data_pipelines.soil_moisture_monthly_awo.get_ncss_time_range")
@patch("requests.get")
def test_download_mdb_soilmoisture_subset(mock_get, mock_time_range, mock_config, sample_gdf):
    # Setup
    mock_config.output_path.mkdir(parents=True, exist_ok=True)
    mock_time_range.return_value = "2023-12-01"

    # Mock response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data_chunk"]
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # Ensure local file does not exist
    if mock_config.local_root_zone_soil_moisture_relative_netcdf_path.exists():
        mock_config.local_root_zone_soil_moisture_relative_netcdf_path.unlink()

    download_mdb_soilmoisture_subset(
        mock_config,
        gdf=sample_gdf,
    )

    assert mock_config.local_root_zone_soil_moisture_relative_netcdf_path.exists()
    assert mock_get.called

@patch("data_pipelines.soil_moisture_monthly_awo.exact_extract")
def test_process_polygon_block_lazy(mock_ee, sample_raster, sample_gdf):
    # Mock exact_extract return
    mock_ee.return_value = pd.DataFrame({"UID": ["A", "B"], "mean": [0.5, 0.6]})
    
    # The function expects a 2D raster (time squeezed out)
    raster_2d = sample_raster.isel(time=0)
    
    # Execute delayed function synchronously
    task = process_polygon_block_lazy(raster_2d, sample_gdf, 2020, 1, "UID")
    result = dask.compute(task, scheduler="sync")[0]
    
    assert result[0] == 2020
    assert result[1] == 1
    assert len(result[2]) == 2
    assert result[2][0] == ("A", 0.5)

def test_write_month_parquet(tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    month_results = [
        (2020, 1, [("A", 0.1), ("B", 0.2)]),
        (2020, 1, [("C", 0.3)])
    ]

    task = write_month_parquet(month_results, cache_path, "sm_pct", "UID")
    path = dask.compute(task, scheduler="sync")[0]

    assert Path(path).exists()
    df = pd.read_parquet(path)
    assert len(df) == 3
    assert df.iloc[0]["year"] == 2020
    assert "sm_pct" in df.columns


def test_aggregate_parquets(mock_config):
    cache_path = mock_config.output_path / "cache"
    output_path = mock_config.output_path
    cache_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dummy parquets
    pd.DataFrame(
        {"UID": ["A"], "sm_pct": [0.1], "year": [2020], "month": [1]}
    ).to_parquet(cache_path / "soil_moisture_2020_01.parquet")
    pd.DataFrame(
        {"UID": ["A"], "sm_pct": [0.2], "year": [2021], "month": [1]}
    ).to_parquet(cache_path / "soil_moisture_2021_01.parquet")

    aggregate_monthly_results(cache_path, output_path, "sm_pct", "UID")

    zip_file = output_path / "soil_moisture_2020_2029.zip"
    assert zip_file.exists()

    df = pd.read_csv(zip_file)
    assert len(df) == 2
    assert 2020 in df["year"].values


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
