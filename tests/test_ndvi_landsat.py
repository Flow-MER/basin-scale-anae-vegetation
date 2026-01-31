import sys
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_pipelines.ndvi_landsat_monthly_dea import (
    compute_ndvi,
    zonal_mean,
    create_macro_tiles,
    bbox_to_wgs84,
    check_dask_graph
)

@pytest.fixture
def mock_config():
    """Mock configuration values."""
    with patch("data_pipelines.ndvi_landsat_monthly_dea.config") as mock_cfg:
        mock_cfg.CRS = "EPSG:3577"
        mock_cfg.TILE_PIXELS = 100
        mock_cfg.PIXEL_SIZE = 30
        mock_cfg.MACRO_TILE_FACTOR = 2
        mock_cfg.POLY_UNIQUE_ID = "UID"
        yield mock_cfg

def test_compute_ndvi():
    """Test NDVI calculation formula and edge cases."""
    # Simple case
    red = np.array([0.1, 0.2])
    nir = np.array([0.5, 0.2])
    # NDVI = (NIR - Red) / (NIR + Red + 1e-6)
    # Case 1: (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.666...
    # Case 2: (0.2 - 0.2) / (0.2 + 0.2) = 0 / 0.4 = 0
    
    result = compute_ndvi(red, nir)
    
    assert result[0] == pytest.approx(0.666666, abs=1e-5)
    assert result[1] == pytest.approx(0.0, abs=1e-5)

    # Division by zero protection check (both 0)
    red_zero = np.array([0.0])
    nir_zero = np.array([0.0])
    result_zero = compute_ndvi(red_zero, nir_zero)
    assert np.isfinite(result_zero[0])
    assert result_zero[0] == pytest.approx(0.0, abs=1e-5)

def test_zonal_mean(mock_config):
    """Test zonal statistics aggregation logic."""
    # Setup 2x2 grid
    # Pixel 0,0: ID=1, NDVI=0.8, Clear=True
    # Pixel 0,1: ID=1, NDVI=0.4, Clear=True
    # Pixel 1,0: ID=2, NDVI=0.2, Clear=False
    # Pixel 1,1: ID=0 (Background), NDVI=0.1, Clear=True
    
    ndvi_data = np.array([[0.8, 0.4], [0.2, 0.1]])
    clear_data = np.array([[1, 1], [0, 1]]) # 1=Clear
    tiles = np.array([[1, 1], [2, 0]])
    
    ndvi_da = xr.DataArray(ndvi_data, dims=("y", "x"))
    clear_da = xr.DataArray(clear_data, dims=("y", "x"))
    
    int_to_uid = {1: "PolyA", 2: "PolyB"}
    
    df = zonal_mean(ndvi_da, clear_da, tiles, int_to_uid)
    
    # Check PolyA
    # Mean = (0.8 + 0.4) / 2 = 0.6
    # Count = 2
    # Clear Pixels = 2
    poly_a = df[df["UID"] == "PolyA"].iloc[0]
    assert poly_a["ndvi"] == pytest.approx(0.6)
    assert poly_a["count"] == 2
    assert poly_a["clear_pixels"] == 2
    
    # Check PolyB
    # Mean = 0.2 (NDVI is computed even if not clear in this specific function, 
    # masking happens before calling zonal_mean in the pipeline, but zonal_mean 
    # calculates 'clear_pixels' count independently)
    poly_b = df[df["UID"] == "PolyB"].iloc[0]
    assert poly_b["ndvi"] == pytest.approx(0.2)
    assert poly_b["count"] == 1
    assert poly_b["clear_pixels"] == 0

def test_create_macro_tiles(mock_config):
    """Test grouping of tiles into macro tiles."""
    # Config: TILE_PIXELS=100, PIXEL_SIZE=30 -> Tile size = 3000m
    # MACRO_TILE_FACTOR=2 -> Macro size = 6000m (2x2 tiles)
    
    # Create 5 tiles
    tiles = [
        {"tile_id": 0, "bounds": (0, 0, 3000, 3000)},       # Bottom-Left
        {"tile_id": 1, "bounds": (3000, 0, 6000, 3000)},    # Bottom-Right
        {"tile_id": 2, "bounds": (0, 3000, 3000, 6000)},    # Top-Left
        {"tile_id": 3, "bounds": (3000, 3000, 6000, 6000)}, # Top-Right
        {"tile_id": 4, "bounds": (6000, 0, 9000, 3000)},    # Outside first macro block
    ]
    
    # Mock bbox_to_wgs84 to avoid ODC dependency issues
    with patch("data_pipelines.ndvi_landsat_monthly_dea.bbox_to_wgs84", return_value=[0,0,1,1]):
        macro_tiles = create_macro_tiles(tiles)
    
    assert len(macro_tiles) >= 2
    
    # Check first macro tile (should contain 0, 1, 2, 3)
    m0 = [m for m in macro_tiles if m["macro_id"] == 0][0]
    m0_tile_ids = sorted([t["tile_id"] for t in m0["tiles"]])
    assert m0_tile_ids == [0, 1, 2, 3]
    
    # Check second macro tile (should contain 4)
    m1 = [m for m in macro_tiles if 4 in [t["tile_id"] for t in m["tiles"]]][0]
    assert len(m1["tiles"]) == 1
    assert m1["tiles"][0]["tile_id"] == 4

def test_bbox_to_wgs84(mock_config):
    """Test coordinate transformation wrapper."""
    with patch("data_pipelines.ndvi_landsat_monthly_dea.Geometry") as MockGeometry:
        mock_geom_instance = MockGeometry.return_value
        mock_geom_wgs84 = mock_geom_instance.to_crs.return_value
        mock_geom_wgs84.boundingbox.left = 140
        mock_geom_wgs84.boundingbox.bottom = -40
        mock_geom_wgs84.boundingbox.right = 150
        mock_geom_wgs84.boundingbox.top = -30
        
        bbox = [1000, 2000, 3000, 4000]
        result = bbox_to_wgs84(bbox)
        
        assert result == [140, -40, 150, -30]
        MockGeometry.assert_called()
        mock_geom_instance.to_crs.assert_called_with("EPSG:4326")

def test_check_dask_graph():
    """Test dask graph size safety check."""
    # Mock object with __dask_graph__
    class MockDaskObj:
        def __dask_graph__(self):
            return range(100) # 100 tasks
            
    obj = MockDaskObj()
    
    # Should pass if limit > 100
    check_dask_graph(obj, max_tasks=200)
    
    # Should fail if limit < 100
    with pytest.raises(RuntimeError, match="FATAL"):
        check_dask_graph(obj, max_tasks=50)

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))