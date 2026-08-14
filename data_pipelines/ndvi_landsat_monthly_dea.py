"""
DEA Monthly NDVI Processor - Macro-Region Loading Strategy
===========================================================

Optimized for large-scale processing of Murray-Darling Basin for all times in Landsat data cube (1M km², 480 months).

MACRO-REGION STRATEGY:
- Divide study area into macro-tiles (8192×8192 pixels = 245km × 245km)
- STAC search ONCE per month for full extent
- Load Landsat/WOfS data per macro-tile with .persist()
- Process all tiles within macro-tile via spatial slicing (no redundant loads)
- Explicit memory cleanup after each macro-tile

SCALABILITY:
- 1M km² ≈ 16 macro-tiles
- 480 months × 16 macro-tiles = 7,680 load operations (vs 120,000 per-tile loads)
- 98% reduction in network I/O
"""

import calendar
import gc
import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import psutil
import rasterio
from dask import compute as dask_compute
from dask import delayed
from dask.distributed import Future, get_client
from odc.geo.crs import CRS as GeoCRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.geom import box as geo_box
from pystac_client import Client
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box
from tqdm import tqdm

# Add project root to sys.path to allow imports from config.py and tools/
# This handles cases where the script is moved to a subfolder (e.g., input_pipelines/)
current_path = Path(__file__).resolve().parent
if (current_path / "config.py").exists():
    project_root = current_path
else:
    project_root = current_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.dask import start_dask
from tools.logging_setup import setup_logging

###################################################
# STACCache is authors internal infrastructure
# this handles other environments
try:
    from tools.stac_cache import STACCache
except ImportError:
    STACCache = None


def _noop_decorator(func):
    return func

_instance = STACCache.get_instance() if STACCache else None
cache_search = _instance.cache_search if _instance else _noop_decorator
###################################################


from config import NDVILandsatConfig, load_config

logger = logging.getLogger(__name__)

# silence noisy warning from rasterio during dask startup
import warnings

from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings(
    "ignore", message="Dataset has no geotransform", category=NotGeoreferencedWarning
)


def validate_config(config) -> None:
    """
    Validates configuration before processing starts.
    Catches issues early rather than failing hours into a batch job.
    """
    errors = []
    warnings = []

    # 1. Check critical paths exist
    if not config.shapefile_path.is_file():
        errors.append(f"shapefile_path is not a file: {config.shapefile_path}")

    # 2. Check value ranges
    if config.pixel_size <= 0:
        errors.append(f"pixel_size must be positive, got: {config.pixel_size}")

    if config.tile_pixels < 256:
        warnings.append(
            f"tile_pixels is very small ({config.tile_pixels}), which may increase overhead."
        )

    if config.macro_tile_factor < 1:
        errors.append(f"macro_tile_factor must be >= 1, got: {config.macro_tile_factor}")

    # 3. Check STAC URL
    if not config.stac_url.startswith("http"):
        errors.append(f"stac_url must be a valid URL, got: {config.stac_url}")

    # Report results
    if warnings:
        for w in warnings:
            logger.warning(f"Config warning: {w}")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


# =========================
# PATH MANAGEMENT
# =========================
def get_tiles_path(config):
    return config.output_path / "tile_masks"


def get_mapping_file(config: NDVILandsatConfig):
    return get_tiles_path(config) / "uid_mapping.json"


# =========================
# SPATIAL GRID
# =========================
# Global variable on each worker to store the mapping once loaded
_WORKER_UID_CACHE = None


def get_mapping_on_worker(config):
    """
    Lazy-loads only the int_to_uid mapping into worker memory.
    """
    global _WORKER_UID_CACHE
    if _WORKER_UID_CACHE is None:
        mapping_file = get_mapping_file(config).resolve()

        try:
            with open(mapping_file) as f:
                data = json.load(f)
                _WORKER_UID_CACHE = {row[0]: row[1] for row in data["polygons"]}
                logger.debug(f"Worker loaded {len(_WORKER_UID_CACHE)} UIDs from {mapping_file}")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Worker failed to load mapping from {mapping_file}: {e}") from e

    return _WORKER_UID_CACHE


def get_global_uid_mapping(config):
    """Load existing UID mapping. Must be created during rasterization."""
    mapping_file = get_mapping_file(config)
    if not mapping_file.exists():
        raise RuntimeError(
            f"UID mapping file {mapping_file} does not exist. "
            "Run rasterization first to generate tiles and mapping together."
        )

    try:
        with open(mapping_file) as f:
            data = json.load(f)
            polygons = data["polygons"]  # [[int, uid, pixel_count], ...]
            uid_to_int = {row[1]: row[0] for row in polygons}
            int_to_uid = {row[0]: row[1] for row in polygons}
            logger.debug(f"Loaded UID mapping: {len(polygons)} polygons")
            return uid_to_int, int_to_uid
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        raise RuntimeError(f"Corrupted UID mapping file {mapping_file}: {e}") from e


def compute_tile_hash(tile_path):
    """Compute SHA256 hash of tile file for validation."""
    sha256 = hashlib.sha256()
    with open(tile_path, "rb") as f:
        sha256.update(f.read())
    return sha256.hexdigest()[:16]


def rasterize_tile_polygons(
    tile_id,
    tile_bounds,
    polygons,
    uid_to_int,
    total_pixels,
    tile_hashes,
    config: NDVILandsatConfig,
):
    """
    Rasterizes polygons within a specific tile and updates pixel counts.

    Args:
        tile_id (int): Unique identifier for the tile.
        tile_bounds (tuple): (minx, miny, maxx, maxy) bounds of the tile.
        polygons (GeoDataFrame): The source polygons.
        uid_to_int (dict): Mapping from UID string to integer ID.
        total_pixels (dict): Dictionary to update with pixel counts per UID.
        tile_hashes (dict): Dictionary to store hash of generated tile file.
        config (NDVILandsatConfig): Configuration object.
    """
    minx, miny, maxx, maxy = tile_bounds
    tile_geom = box(minx, miny, maxx, maxy)
    polys = polygons[polygons.intersects(tile_geom)]

    if polys.empty:
        return

    tiles_file = get_tiles_path(config) / f"tile_{tile_id:05d}.tif"
    tiles_file.parent.mkdir(parents=True, exist_ok=True)

    transform = from_bounds(minx, miny, maxx, maxy, config.tile_pixels, config.tile_pixels)
    polys_mapped = polys.copy()
    polys_mapped["raster_id"] = polys_mapped[config.poly_unique_id].map(uid_to_int)

    shapes = zip(polys_mapped.geometry, polys_mapped["raster_id"], strict=True)
    tiles = rasterize(
        shapes,
        out_shape=(config.tile_pixels, config.tile_pixels),
        transform=transform,
        fill=0,
        dtype="int32",
    )

    unique, counts = np.unique(tiles[tiles > 0], return_counts=True)
    uid_to_string = polys_mapped.set_index("raster_id")[config.poly_unique_id].to_dict()
    for uid_int, count in zip(unique, counts, strict=True):
        uid = uid_to_string[uid_int]
        total_pixels[uid] = total_pixels.get(uid, 0) + count

    with rasterio.open(
        tiles_file,
        "w",
        driver="GTiff",
        height=config.tile_pixels,
        width=config.tile_pixels,
        count=1,
        dtype="int32",
        crs=config.crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(tiles, 1)

    tile_hashes[str(tile_id)] = compute_tile_hash(tiles_file)


def load_or_create_raster_tiles(config):
    """
    Loads existing raster tiles or generates them from the polygon shapefile.

    Validates existing tiles against stored hashes. If validation fails or tiles
    are missing, regenerates the entire tiling scheme.

    Args:
        config (NDVILandsatConfig): Configuration object.

    Returns:
        list: List of dictionaries containing tile metadata (id, bounds).
    """
    tiles_path = get_tiles_path(config)
    mapping_file = get_mapping_file(config)

    if tiles_path.exists() and mapping_file.exists():
        try:
            with open(mapping_file) as f:
                data = json.load(f)
                expected_hashes = data.get("tile_hashes", {})

            tiles = []
            mismatched = []

            for f in tiles_path.glob("tile_*.tif"):
                tile_id = int(f.stem.split("_")[1])
                tile_id_str = str(tile_id)

                if tile_id_str in expected_hashes:
                    if compute_tile_hash(f) != expected_hashes[tile_id_str]:
                        mismatched.append(tile_id)
                        continue

                with rasterio.open(f) as src:
                    tiles.append(
                        {
                            "tile_id": tile_id,
                            "bounds": (
                                src.bounds.left,
                                src.bounds.bottom,
                                src.bounds.right,
                                src.bounds.top,
                            ),
                        }
                    )

            found_ids = {t["tile_id"] for t in tiles}
            expected_ids = {int(k) for k in expected_hashes.keys()}
            missing = expected_ids - found_ids

            if missing or mismatched:
                logger.warning(f"Missing: {len(missing)}, Corrupted: {len(mismatched)} tiles")
                logger.info("Regenerating all tiles")
            else:
                logger.info(f"Validated {len(tiles)} tiles")
                return sorted(tiles, key=lambda x: x["tile_id"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Validation failed: {e}")

    logger.info("Generating raster tiles and UID mapping...")
    if tiles_path.exists():
        shutil.rmtree(tiles_path)
    tiles_path.mkdir(parents=True)
    mapping_file.unlink(missing_ok=True)

    polygons = gpd.read_file(config.shapefile_path).to_crs(config.crs)

    unique_uids = sorted(polygons[config.poly_unique_id].unique())
    uid_to_int = {uid: i + 1 for i, uid in enumerate(unique_uids)}

    minx, miny, maxx, maxy = polygons.total_bounds
    tile_size = config.tile_pixels * config.pixel_size
    xs = np.arange(
        np.floor(minx / config.pixel_size) * config.pixel_size,
        np.ceil(maxx / config.pixel_size) * config.pixel_size,
        tile_size,
    )
    ys = np.arange(
        np.floor(miny / config.pixel_size) * config.pixel_size,
        np.ceil(maxy / config.pixel_size) * config.pixel_size,
        tile_size,
    )

    total_pixels = {}
    tile_hashes = {}
    tile_coords = [(x, y) for x in xs for y in ys]
    for tile_id, (x, y) in enumerate(tqdm(tile_coords, desc="Rasterising tiles")):
        rasterize_tile_polygons(
            tile_id,
            (x, y, x + tile_size, y + tile_size),
            polygons,
            uid_to_int,
            total_pixels,
            tile_hashes,
            config,
        )
    logger.info("Raster tile generation complete.")

    # Build polygon data: [int_value, uid, pixel_count]
    polygon_data = [[uid_to_int[uid], uid, int(total_pixels.get(uid, 0))] for uid in unique_uids]

    with open(mapping_file, "w") as f:
        json.dump({"polygons": polygon_data, "tile_hashes": tile_hashes}, f, indent=2)
    logger.info(
        f"Created UID mapping: {len(polygon_data)} polygons, {len(tile_hashes)} tiles ({config.pixel_size}m)"
    )

    return load_or_create_raster_tiles(config)


def create_macro_tiles(raster_tiles, config):
    """Group tiles into macro-regions (8192×8192 pixels)."""
    if not raster_tiles:
        return []

    # Get tile size from first tile bounds
    first_tile = raster_tiles[0]["bounds"]
    tile_size_m = first_tile[2] - first_tile[0]  # maxx - minx in meters
    macro_size_m = tile_size_m * config.macro_tile_factor

    all_bounds = [t["bounds"] for t in raster_tiles]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    macro_tiles = []
    macro_id = 0

    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            macro_bbox = (x, y, min(x + macro_size_m, maxx), min(y + macro_size_m, maxy))

            tiles_in_macro = []
            for tile in raster_tiles:
                tx0, ty0, tx1, ty1 = tile["bounds"]
                mx0, my0, mx1, my1 = macro_bbox
                if not (tx1 <= mx0 or tx0 >= mx1 or ty1 <= my0 or ty0 >= my1):
                    tiles_in_macro.append(tile)

            if tiles_in_macro:
                macro_tiles.append(
                    {
                        "macro_id": macro_id,
                        "bounds": macro_bbox,
                        "bounds_wgs84": bbox_to_wgs84(macro_bbox, config),
                        "geobox": GeoBox.from_bbox(
                            macro_bbox, GeoCRS(config.crs), resolution=config.pixel_size
                        ),
                        "tiles": tiles_in_macro,
                    }
                )
                macro_id += 1

            x += macro_size_m
        y += macro_size_m
    tile_size_px = int(tile_size_m / config.pixel_size)
    macro_size_px = tile_size_px * config.macro_tile_factor
    logger.info(
        f"{len(raster_tiles)} tiles ({tile_size_px}×{tile_size_px} = {tile_size_m / 1000:.1f}km²) grouped into {len(macro_tiles)} macro-tiles ({macro_size_px}×{macro_size_px} = {macro_size_m / 1000:.1f}km²)"
    )
    return macro_tiles


# =========================
# UTILITIES
# =========================


def check_dask_graph(obj, config, name="object", max_tasks=None, max_partitions=None):
    """Monitor Dask graph size and raise error if thresholds exceeded."""
    if max_tasks is None:
        max_tasks = config.max_dask_tasks
    if max_partitions is None:
        max_partitions = config.max_dask_partitions

    try:
        if hasattr(obj, "__dask_graph__"):
            n_tasks = len(obj.__dask_graph__())
            logger.debug(f"{name}: {n_tasks:,} tasks")
            if n_tasks > max_tasks:
                raise RuntimeError(
                    f"FATAL: {name} has {n_tasks:,} tasks (>{max_tasks:,} threshold). "
                    f"Graph explosion detected. Call .persist() earlier or rechunk data."
                )

        if hasattr(obj, "data") and hasattr(obj.data, "npartitions"):
            n_parts = obj.data.npartitions
            logger.debug(f"{name}: {n_parts} partitions")
            if n_parts > max_partitions:
                raise RuntimeError(
                    f"FATAL: {name} has {n_parts} partitions (>{max_partitions} threshold). "
                    f"Too fragmented. Rechunk with larger chunks."
                )
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Could not check {name}: {e}")


def bbox_to_wgs84(bbox, config):
    """Converts a bounding box from the project CRS to WGS84 (EPSG:4326)."""
    for i, x in enumerate(bbox):
        if not np.isfinite(x):
            raise ValueError(f"Invalid bbox coordinate at index {i}: {x} in {bbox}")
    geom = Geometry(geo_box(*bbox, GeoCRS(config.crs)))
    geom_wgs84 = geom.to_crs("EPSG:4326").boundingbox
    return [geom_wgs84.left, geom_wgs84.bottom, geom_wgs84.right, geom_wgs84.top]


def stac_client(config):
    """Initializes and returns a PySTAC Client for the configured STAC URL."""
    try:
        # Change 'request_session' to 'session'
        return Client.open(config.stac_url)
    except Exception as e:
        logger.error(f"Failed to connect to STAC catalog at {config.stac_url}: {e}")
        raise


def log_memory():
    """Log current memory usage."""
    mem = psutil.virtual_memory()
    return f"{mem.percent:.1f}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB)"


def load_tile_mask(tile_id, config):
    """Loads the raster mask for a specific tile ID from disk."""
    tiles_file = get_tiles_path(config) / f"tile_{tile_id:05d}.tif"
    if tiles_file.exists():
        with rasterio.open(tiles_file) as src:
            return src.read(1)
    return None


def compute_ndvi(red, nir):
    """Computes Normalized Difference Vegetation Index (NDVI)."""
    return (nir - red) / (nir + red + 1e-6)


def zonal_mean(ndvi, clear_mask, tiles, int_to_uid, config):
    """
    Computes zonal statistics (mean NDVI, counts) for polygons within a tile.

    Args:
        ndvi (xarray.DataArray): NDVI data for the tile.
        clear_mask (xarray.DataArray): Boolean mask of clear pixels.
        tiles (numpy.ndarray): Rasterized polygon IDs for the tile.
        int_to_uid (dict): Mapping from integer raster IDs to string UIDs.
        config (NDVILandsatConfig): Configuration object.

    Returns:
        pd.DataFrame: Zonal statistics for the tile, or None if no valid data.
    """
    nd = ndvi.values
    cm = clear_mask.values
    lb = tiles

    valid_poly = (lb > 0) & (~np.isnan(nd))

    if not np.any(valid_poly):
        return None

    nd_valid = nd[valid_poly]
    lb_valid = lb[valid_poly]

    unique_ids, inverse, counts = np.unique(lb_valid, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=nd_valid) / counts

    # Validate NDVI range
    if np.any((means < -1.1) | (means > 1.1)):
        logger.warning(
            f"NDVI values outside expected range [-1, 1]: min={means.min():.3f}, max={means.max():.3f}"
        )

    cm_valid = cm[valid_poly]
    clear_counts = np.bincount(inverse, weights=cm_valid.astype(int))

    string_uids = [int_to_uid.get(int_id) for int_id in unique_ids]
    valid_mask = [u is not None for u in string_uids]

    return pd.DataFrame(
        {
            config.poly_unique_id: [u for u in string_uids if u is not None],
            "ndvi": means[valid_mask],
            "count": counts[valid_mask],
            "clear_pixels": clear_counts[valid_mask],
        }
    )


@cache_search
def search_stac_collection(catalog, collections, bbox, start_date, end_date, description):
    """Helper to search STAC catalog with error handling."""
    logger.info(f"  Searching for {description} data {start_date} to {end_date}...")
    try:
        items = catalog.search(
            collections=collections,
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            limit=300,
        ).item_collection()
    except Exception as e:
        logger.error(f"  {description} STAC search failed: {e}")
        return None

    if len(items) == 0:
        logger.info(f"  No {description} data for {start_date[:7]}")
        return None

    logger.info(f"  - Found {len(items)} {description} items")
    return items


def load_landsat_macro(items, geobox, macro_id, year, month, config):
    """Loads Landsat data, computes NDVI and clear mask for a macro-tile."""
    logger.debug(f"   Macro-tile {macro_id}: loading landsat data...")

    bands = ["nbart_red", "nbart_nir", "oa_fmask"]

    patch_url = STACCache.get_instance().patch_url if STACCache else None

    try:
        landsat_ds = odc.stac.load(
            items,
            bands=bands,
            geobox=geobox,
            groupby="solar_day",
            chunks={"time": 1, "x": config.tile_pixels, "y": config.tile_pixels},
            fail_on_error=True,
            patch_url=patch_url,
        )
    except Exception as e:
        logger.error(f"Landsat load failed for macro-tile {macro_id}: {e}")
        raise

    if (
        landsat_ds is None
        or landsat_ds.time.size == 0
        or landsat_ds.sizes["x"] == 0
        or landsat_ds.sizes["y"] == 0
    ):
        logger.warning(f"      Landsat: no data for macro-tile {macro_id} ({year}-{month:02d})")
        return None, None

    logger.info(
        f"   Loading {landsat_ds.time.size}/{len(items)} Landsat scenes for macro-tile {macro_id}..."
    )

    landsat_ds = landsat_ds.persist()
    check_dask_graph(landsat_ds, config, "landsat_ds after persist")
    logger.debug(f"      Memory after Landsat load: {log_memory()}")

    # Clear mask & NDVI
    clear_mask = ((landsat_ds.oa_fmask == 1) | (landsat_ds.oa_fmask == 5)).persist()

    if clear_mask.isnull().all():
        logger.info(f"      Macro-tile {macro_id}: no clear pixels ({year}-{month:02d})")
        return None, None

    landsat_ds["nbart_red"] = landsat_ds.nbart_red.where(clear_mask)
    landsat_ds["nbart_nir"] = landsat_ds.nbart_nir.where(clear_mask)

    ndvi = compute_ndvi(landsat_ds.nbart_red, landsat_ds.nbart_nir).persist()
    del landsat_ds
    logger.debug(f"      Memory after NDVI compute: {log_memory()}")
    return ndvi, clear_mask


def apply_wofs_mask(ndvi, wofs_items, geobox, macro_id, config):
    """Loads WOfS data and masks open water from NDVI."""
    logger.debug(f"   Macro-tile {macro_id}: loading WOfS data and open water mask")

    patch_url = STACCache.get_instance().patch_url if STACCache else None

    try:
        wofs_data = odc.stac.load(
            wofs_items,
            bands=["water"],
            geobox=geobox,
            groupby="solar_day",
            chunks={"time": 1, "x": config.tile_pixels, "y": config.tile_pixels},
            fail_on_error=False,
            patch_url=patch_url,
        )
        if wofs_data is not None and "water" in wofs_data and wofs_data.time.size > 0:
            logger.info(
                f"   Loading {wofs_data.time.size}/{len(wofs_items)} WOfS scenes for macro-tile {macro_id}..."
            )
            wofs_data = wofs_data.persist()
            water_int = wofs_data.water.fillna(0).astype("uint8")
            quality_mask = (water_int & 0b01100011) == 0
            open_water = (water_int & (1 << 7)) > 0
            vegetation_mask = ~open_water & quality_mask

            ndvi_masked = ndvi.where(vegetation_mask)

            masked_pixels = vegetation_mask.sum().compute()
            logger.debug(f"      WOfS applied, {masked_pixels:.0f} open water pixels masked")
            del wofs_data, vegetation_mask, water_int
            return ndvi_masked
    except Exception as e:
        logger.warning(f"WOfS load or masking failed: {e}", exc_info=True)

    return ndvi


# =========================
# MACRO-REGION PROCESSING
# =========================


def cache_empty_tile(cache_file, config):
    """Writes an empty DataFrame to the cache file to prevent reprocessing."""
    df = pd.DataFrame(
        {
            config.poly_unique_id: pd.Series(dtype="object"),
            "ndvi": pd.Series(dtype="float64"),
            "count": pd.Series(dtype="int32"),
            "clear_pixels": pd.Series(dtype="int32"),
        }
    )
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    except Exception as e:
        logger.warning(f"Failed to write empty cache {cache_file}: {e}")
    return df


@delayed
def process_tile_from_macro(tile, ndvi_med, clear_mask_med, tile_masks, config):
    """
    Worker-level processing: loads mapping from local cache and computes zonal stats.
    Processes a single sub-tile via spatial slicing from macro-region data.
    """
    # Retrieve the mapping from the local process memory
    int_to_uid = get_mapping_on_worker(config)

    tile_id = tile["tile_id"]
    cache_file = (
        config.output_path
        / "cache"
        / f"{ndvi_med.attrs['year']}_{ndvi_med.attrs['month']:02d}"
        / f"tile_{tile_id:05d}.parquet"
    )

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    # Resolve Future if needed (when called via dask.delayed)
    if isinstance(tile_masks, Future):
        tile_masks = tile_masks.result()

    tiles_mask = tile_masks.get(tile_id)
    if tiles_mask is None:
        logger.debug(f"Tile {tile_id} has no mask, skipping")
        return cache_empty_tile(cache_file, config)

    tx0, ty0, tx1, ty1 = tile["bounds"]

    try:
        tile_ndvi = ndvi_med.sel(x=slice(tx0, tx1), y=slice(ty1, ty0))
        tile_clear = clear_mask_med.sel(x=slice(tx0, tx1), y=slice(ty1, ty0))
    except (KeyError, ValueError):
        logger.debug(f"Tile {tile_id} has no data, skipping")
        return cache_empty_tile(cache_file, config)

    if tile_ndvi.sizes.get("x", 0) == 0 or tile_ndvi.sizes.get("y", 0) == 0:
        logger.debug(f"Tile {tile_id} has no data (x or y dimension zero length), skipping")
        return cache_empty_tile(cache_file, config)

    if tile_ndvi.isnull().all():
        logger.debug(f"Tile {tile_id} fully masked, skipping")
        return cache_empty_tile(cache_file, config)

    result = zonal_mean(tile_ndvi, tile_clear, tiles_mask, int_to_uid, config)
    del tile_ndvi, tile_clear, tiles_mask

    if result is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Failed to cache tile {tile_id}: {e}")
            cache_file.unlink(missing_ok=True)
        return result

    return cache_empty_tile(cache_file, config)


def process_macro_tile(macro_tile, year, month, items, wofs_items, config):
    """
    Process a macro-tile (8192x8192) by loading Landsat/NDVI once, masking with WOfS,
    aggregating to monthly median, and processing all sub-tiles (2048x2048) sequentially.
    Cached tile results are read if available.
    """

    macro_id = macro_tile["macro_id"]
    tiles = macro_tile["tiles"]
    geobox = macro_tile["geobox"]

    cache_path = config.output_path / "cache" / f"{year}_{month:02d}"
    cached_tiles = []
    uncached_tiles = []

    for tile in tiles:
        cache_file = cache_path / f"tile_{tile['tile_id']:05d}.parquet"
        if cache_file.exists():
            try:
                cached_tiles.append(pd.read_parquet(cache_file))
            except Exception as e:
                logger.warning(
                    f"Failed to read cached tile {tile['tile_id']:05d}, will regenerate: {e}"
                )
                uncached_tiles.append(tile)
        else:
            uncached_tiles.append(tile)

    if not uncached_tiles:
        logger.info(f"   Macro-tile {macro_id} fully cached ({len(tiles)} sub-tiles)")
        return cached_tiles

    logger.info(
        f"   Macro-tile {macro_id}: processing {len(uncached_tiles)}/{len(tiles)} sub-tiles"
    )

    tile_masks = {
        tile["tile_id"]: load_tile_mask(tile["tile_id"], config) for tile in uncached_tiles
    }
    tile_masks = {k: v for k, v in tile_masks.items() if v is not None}

    dask_client = get_client()
    tile_masks_future = dask_client.scatter(tile_masks, broadcast=True)

    ndvi, clear_mask = load_landsat_macro(items, geobox, macro_id, year, month, config)
    if ndvi is None:
        logger.info(
            f"      Macro-tile {macro_id}: No data found. Caching empty results for {len(uncached_tiles)} tiles."
        )
        for tile in uncached_tiles:
            cache_file = cache_path / f"tile_{tile['tile_id']:05d}.parquet"
            cached_tiles.append(cache_empty_tile(cache_file, config))
        return cached_tiles

    if wofs_items:
        ndvi = apply_wofs_mask(ndvi, wofs_items, geobox, macro_id, config)
    else:
        logger.debug("WOfS data unavailable or empty")

    if ndvi.isnull().all():
        logger.info(f"      Macro-tile {macro_id}: NDVI fully masked ({year}-{month:02d})")
        logger.info(
            f"      Macro-tile {macro_id}: NDVI fully masked ({year}-{month:02d}). Caching empty results."
        )
        for tile in uncached_tiles:
            cache_file = cache_path / f"tile_{tile['tile_id']:05d}.parquet"
            cached_tiles.append(cache_empty_tile(cache_file, config))
        return cached_tiles

    # -------------------------
    # Monthly median & clear mask aggregation
    # -------------------------
    ndvi_med = ndvi.median("time").where(lambda x: x > 0.0)
    ndvi_med.attrs["year"] = year
    ndvi_med.attrs["month"] = month
    ndvi_med = ndvi_med.persist()
    del ndvi
    check_dask_graph(ndvi_med, config, "ndvi_med after persist")

    clear_mask_med = clear_mask.max("time")
    clear_mask_med = clear_mask_med.persist()

    # -------------------------
    # Cleanup macro-tile-level Dask objects
    # -------------------------

    del clear_mask

    tile_tasks = []
    for tile in uncached_tiles:
        task = process_tile_from_macro(tile, ndvi_med, clear_mask_med, tile_masks_future, config)
        tile_tasks.append(task)

    logger.info(f"   Macro-tile {macro_id}: parallel compute for {len(tile_tasks)} sub-tiles")
    new_results = dask_compute(*tile_tasks)

    del ndvi_med, clear_mask_med
    return cached_tiles + [r for r in new_results if r is not None]


def process_one_month(year, month, raster_tiles, macro_tiles, catalog, config):
    """Process month using macro-region strategy."""
    final_out = config.output_path / f"NDVI_{year}_{month:02d}.parquet"
    if final_out.exists():
        logger.info(
            f"  Found saved result {year}-{month:02d} in {final_out.name} - skipping processing"
        )
        return

    logger.info(f"Processing {year}-{month:02d}")

    all_bounds = [t["bounds"] for t in raster_tiles]
    full_bbox = [
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    ]
    bbox_wgs84 = bbox_to_wgs84(full_bbox, config)

    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"

    items = search_stac_collection(
        catalog, config.landsat_collections, bbox_wgs84, start, end, "Landsat"
    )
    if not items:
        return

    wofs_items = search_stac_collection(
        catalog, [config.wofs_collection], bbox_wgs84, start, end, "WOfS"
    )
    if not wofs_items:
        return

    # Pre-cache all Landsat + WOfS assets for this month in one parallel burst.
    # All macro_tiles draw from this same pool, so preloading here saturates
    # the connection once rather than doing 16-20 smaller downloads in the loop.

    # Calculate WGS84 bboxes ordered by macro-tile processing sequence
    # This allows the cache to prioritize downloads for the first macro-tiles
    macro_bboxes = [m["bounds_wgs84"] for m in macro_tiles]

    landsat_bands = ["nbart_red", "nbart_nir", "oa_fmask"]

    # Fire and forget — downloads run in the background while the loop starts
    # preload_future_landsat = cache._executor.submit(cache.cache_items, items, landsat_bands, tile_bboxes)
    # preload_future_wofs = cache._executor.submit(cache.cache_items, wofs_items, ["water"], tile_bboxes)
    # Fire-and-forget — submit from MAIN THREAD
    if STACCache:
        cache = STACCache.get_instance()
        cache.submit_batches(
            [(items, landsat_bands), (wofs_items, ["water"])],
            intersection_filter=macro_bboxes,
        )

    # Ensure Dask is running
    try:
        dask_client = get_client()
    except (ValueError, OSError):
        dask_client = start_dask()

    # Gathers results one macro-region at a time
    all_results = []
    for macro_tile in macro_tiles:
        logger.info(
            f"Macro-tile {macro_tile['macro_id']} - {year}-{month:02d} - memory use: {log_memory()}"
        )

        # Retry logic to handle worker crashes (e.g. OOM) or lost scattered data
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = process_macro_tile(macro_tile, year, month, items, wofs_items, config)
                if res:
                    all_results.extend(res)
                break
            except Exception as e:
                logger.warning(
                    f"Macro-tile {macro_tile['macro_id']} failed attempt {attempt + 1}/{max_retries}: {e}"
                )
                dask_client.run(gc.collect)
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)

        # FORCE CLEANUP: Tell all workers to clear memory before next macro-tile
        dask_client.run(gc.collect)
    # preload_future_landsat.result()
    # preload_future_wofs.result()

    # Filter empty DataFrames to prevent FutureWarning in pd.concat
    valid_results = [df for df in all_results if not df.empty]

    if not valid_results:
        logger.info(f"  No results for {year}-{month:02d}. Creating empty baseline.")
        combined = pd.DataFrame(
            {
                config.poly_unique_id: pd.Series(dtype="object"),
                "ndvi": pd.Series(dtype="float64"),
                "count": pd.Series(dtype="int32"),
                "clear_pixels": pd.Series(dtype="int32"),
                "w_ndvi": pd.Series(dtype="float64"),
            }
        )
    else:
        combined = pd.concat(valid_results, ignore_index=True)
        combined["w_ndvi"] = combined["ndvi"] * combined["count"]

    aggregated = (
        combined.groupby(config.poly_unique_id)
        .agg(
            w_ndvi_sum=("w_ndvi", "sum"),
            count_sum=("count", "sum"),
            clear_pixels_sum=("clear_pixels", "sum"),
        )
        .reset_index()
    )

    mapping_file = get_mapping_file(config)
    with open(mapping_file) as f:
        data = json.load(f)
    all_uids_df = pd.DataFrame(
        data["polygons"], columns=["int_value", config.poly_unique_id, "total_pixels"]
    )
    all_uids_df = all_uids_df[
        [config.poly_unique_id, "total_pixels"]
    ]  # Keep only UID and pixel count

    final_result = all_uids_df.merge(aggregated, on=config.poly_unique_id, how="left")

    final_result["w_ndvi_sum"] = final_result["w_ndvi_sum"].fillna(0)
    final_result["count_sum"] = final_result["count_sum"].fillna(0)
    final_result["clear_pixels_sum"] = final_result["clear_pixels_sum"].fillna(0)

    final_result["ndvi"] = np.where(
        final_result["count_sum"] > 0,
        final_result["w_ndvi_sum"] / final_result["count_sum"],
        np.nan,
    )

    final_result["quality"] = (
        final_result["clear_pixels_sum"] / final_result["total_pixels"]
    ).fillna(0)

    final_result["year"] = year
    final_result["month"] = month

    final_result = final_result.rename(
        columns={"count_sum": "count", "clear_pixels_sum": "clear_pixels"}
    )

    final_result = final_result.drop(columns=["w_ndvi_sum"])
    try:
        tmp_out = final_out.with_suffix(".tmp.parquet")
        final_result.to_parquet(tmp_out)
        tmp_out.rename(final_out)
        logger.info(f"  Completed {year}-{month:02d}: {len(final_result)} polygons")
    except Exception as e:
        logger.error(f"  Failed to write final output for {year}-{month:02d}: {e}")
        tmp_out.unlink(missing_ok=True)
        raise


# =========================
# MAIN
# =========================


def main():
    """Main execution entry point for Landsat NDVI processing."""
    config = load_config("ndvi_landsat")
    validate_config(config)
    setup_logging(config.log_path, "ndvi_processing")
    # create output directory
    try:
        config.output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Cannot create output_dir {config.output_path}: {e}") from e

    # -----------------------------------------------------------------
    # prepare pipeline
    # -----------------------------------------------------------------

    raster_tiles = load_or_create_raster_tiles(config)
    macro_tiles = create_macro_tiles(raster_tiles, config)

    if config.end_date is None:
        current_date = datetime.now()
        end_year, end_month = current_date.year, current_date.month - 1
        if end_month == 0:
            end_year -= 1
            end_month = 12
    else:
        end_year, end_month = config.end_date
    start_year, start_month = config.start_date

    logger.info(f"Processing {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")

    # persistent stac session
    catalog = stac_client(config)
    try:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if year == start_year and month < start_month:
                    continue
                if (year, month) > (end_year, end_month):
                    break
                process_one_month(year, month, raster_tiles, macro_tiles, catalog, config)

                # Previous month's files are on disk and won't be looked up again.
    finally:
        try:
            # close Dask
            get_client().close()
            if STACCache:
                STACCache.get_instance().close()
        except (ValueError, OSError):
            pass


if __name__ == "__main__":
    main()
