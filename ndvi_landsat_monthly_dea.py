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

import os
import gc
import json
import logging
import calendar
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import sys
import hashlib

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

import odc.stac
from odc.geo.crs import CRS as GeoCRS
from odc.geo.geom import Geometry, box as geo_box
from pystac_client import Client

import psutil
import time
from tqdm import tqdm
import dask
from dask.distributed import get_client
from config import ndvi_landsat_cfg as config
from tools.logging_setup import setup_logging
from tools.dask import start_dask

logger = logging.getLogger(__name__)


# =========================
# SPATIAL GRID
# =========================
# Global variable on each worker to store the mapping once loaded
_WORKER_UID_CACHE = None

def get_mapping_on_worker():
    """
    Lazy-loads only the int_to_uid mapping into worker memory.
    """
    global _WORKER_UID_CACHE
    if _WORKER_UID_CACHE is None:
        mapping_file = Path(config.OUTPUT_DIR).resolve() / "uid_mapping.json"
        
        try:
            with open(mapping_file) as f:
                data = json.load(f)
                _WORKER_UID_CACHE = {row[0]: row[1] for row in data["polygons"]}
                logger.debug(f"Worker loaded {len(_WORKER_UID_CACHE)} UIDs from {mapping_file}")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Worker failed to load mapping from {mapping_file}: {e}")
            
    return _WORKER_UID_CACHE

def get_global_uid_mapping():
    """Load existing UID mapping. Must be created during rasterization."""
    mapping_file = config.OUTPUT_DIR / "uid_mapping.json"
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
        raise RuntimeError(f"Corrupted UID mapping file {mapping_file}: {e}")


def compute_tile_hash(tile_path):
    """Compute SHA256 hash of tile file for validation."""
    sha256 = hashlib.sha256()
    with open(tile_path, 'rb') as f:
        sha256.update(f.read())
    return sha256.hexdigest()[:16]


def rasterize_tile_polygons(tile_id, tile_bounds, polygons, uid_to_int, total_pixels, tile_hashes):
    minx, miny, maxx, maxy = tile_bounds
    tile_geom = box(minx, miny, maxx, maxy)
    polys = polygons[polygons.intersects(tile_geom)]

    if polys.empty:
        return

    tiles_file = config.OUTPUT_DIR / "tile_masks" / f"tile_{tile_id:05d}.tif"
    tiles_file.parent.mkdir(parents=True, exist_ok=True)

    transform = from_bounds(
        minx, miny, maxx, maxy, config.TILE_PIXELS, config.TILE_PIXELS
    )
    polys_mapped = polys.copy()
    polys_mapped["raster_id"] = polys_mapped[config.POLY_UID].map(uid_to_int)

    shapes = zip(polys_mapped.geometry, polys_mapped["raster_id"])
    tiles = rasterize(
        shapes,
        out_shape=(config.TILE_PIXELS, config.TILE_PIXELS),
        transform=transform,
        fill=0,
        dtype="int32",
    )

    unique, counts = np.unique(tiles[tiles > 0], return_counts=True)
    uid_to_string = polys_mapped.set_index("raster_id")[config.POLY_UID].to_dict()
    for uid_int, count in zip(unique, counts):
        uid = uid_to_string[uid_int]
        total_pixels[uid] = total_pixels.get(uid, 0) + count

    with rasterio.open(
        tiles_file,
        "w",
        driver="GTiff",
        height=config.TILE_PIXELS,
        width=config.TILE_PIXELS,
        count=1,
        dtype="int32",
        crs=config.CRS,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(tiles, 1)
    
    tile_hashes[str(tile_id)] = compute_tile_hash(tiles_file)


def load_or_create_raster_tiles(polygons_path):
    tiles_dir = config.OUTPUT_DIR / "tile_masks"
    mapping_file = config.OUTPUT_DIR / "uid_mapping.json"

    if tiles_dir.exists() and mapping_file.exists():
        try:
            with open(mapping_file) as f:
                data = json.load(f)
                expected_hashes = data.get("tile_hashes", {})
            
            tiles = []
            mismatched = []
            
            for f in tiles_dir.glob("tile_*.tif"):
                tile_id = int(f.stem.split("_")[1])
                tile_id_str = str(tile_id)
                
                if tile_id_str in expected_hashes:
                    if compute_tile_hash(f) != expected_hashes[tile_id_str]:
                        mismatched.append(tile_id)
                        continue
                
                with rasterio.open(f) as src:
                    tiles.append({
                        "tile_id": tile_id,
                        "bounds": (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                    })
            
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
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True)
    mapping_file.unlink(missing_ok=True)

    polygons = gpd.read_file(polygons_path).to_crs(config.CRS)
    
    unique_uids = sorted(polygons[config.POLY_UID].unique())
    uid_to_int = {uid: i + 1 for i, uid in enumerate(unique_uids)}

    minx, miny, maxx, maxy = polygons.total_bounds
    tile_size = config.TILE_PIXELS * config.PIXEL_SIZE
    xs = np.arange(
        np.floor(minx / config.PIXEL_SIZE) * config.PIXEL_SIZE,
        np.ceil(maxx / config.PIXEL_SIZE) * config.PIXEL_SIZE,
        tile_size,
    )
    ys = np.arange(
        np.floor(miny / config.PIXEL_SIZE) * config.PIXEL_SIZE,
        np.ceil(maxy / config.PIXEL_SIZE) * config.PIXEL_SIZE,
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
        )
    logger.info("Raster tile generation complete.")
    
    # Build polygon data: [int_value, uid, pixel_count]
    polygon_data = [[uid_to_int[uid], uid, int(total_pixels.get(uid, 0))] for uid in unique_uids]
    
    with open(mapping_file, "w") as f:
        json.dump({"polygons": polygon_data, "tile_hashes": tile_hashes}, f, indent=2)
    logger.info(f"Created UID mapping: {len(polygon_data)} polygons, {len(tile_hashes)} tiles ({config.PIXEL_SIZE}m)")

    return load_or_create_raster_tiles(polygons_path)


def create_macro_tiles(raster_tiles):
    """Group tiles into macro-regions (8192×8192 pixels)."""
    if not raster_tiles:
        return []

    # Get tile size from first tile bounds
    first_tile = raster_tiles[0]["bounds"]
    tile_size_m = first_tile[2] - first_tile[0]  # maxx - minx in meters
    macro_size_m = tile_size_m * config.MACRO_TILE_FACTOR

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
                        "bounds_wgs84": bbox_to_wgs84(macro_bbox),
                        "tiles": tiles_in_macro,
                    }
                )
                macro_id += 1

            x += macro_size_m
        y += macro_size_m
    tile_size_px = int(tile_size_m/config.PIXEL_SIZE)
    macro_size_px = tile_size_px * config.MACRO_TILE_FACTOR
    logger.info(f"{len(raster_tiles)} tiles ({tile_size_px}×{tile_size_px} = {tile_size_m/1000:.1f}km²) grouped into {len(macro_tiles)} macro-tiles ({macro_size_px}×{macro_size_px} = {macro_size_m/1000:.1f}km²)")
    return macro_tiles


# =========================
# UTILITIES
# =========================


def check_dask_graph(obj, name="object", max_tasks=None, max_partitions=None):
    """Monitor Dask graph size and raise error if thresholds exceeded."""
    if max_tasks is None:
        max_tasks = config.MAX_DASK_TASKS
    if max_partitions is None:
        max_partitions = config.MAX_DASK_PARTITIONS

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


def bbox_to_wgs84(bbox):
    for i, x in enumerate(bbox):
        if not np.isfinite(x):
            raise ValueError(f"Invalid bbox coordinate at index {i}: {x} in {bbox}")
    geom = Geometry(geo_box(*bbox, GeoCRS(config.CRS)))
    geom_wgs84 = geom.to_crs("EPSG:4326").boundingbox
    return [geom_wgs84.left, geom_wgs84.bottom, geom_wgs84.right, geom_wgs84.top]


def stac_client():
    try:
        # Change 'request_session' to 'session'
        return Client.open(config.STAC_URL)
    except Exception as e:
        logger.error(f"Failed to connect to STAC catalog at {config.STAC_URL}: {e}")
        raise

def log_memory():
    """Log current memory usage."""
    mem = psutil.virtual_memory()
    return f"{mem.percent:.1f}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB)"


def load_tile_mask(tile_id):
    tiles_file = config.OUTPUT_DIR / "tile_masks" / f"tile_{tile_id:05d}.tif"
    if tiles_file.exists():
        with rasterio.open(tiles_file) as src:
            return src.read(1)
    return None


def compute_ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-6)


def zonal_mean(ndvi, clear_mask, tiles, int_to_uid):
    nd = ndvi.values
    cm = clear_mask.values
    lb = tiles

    valid_poly = (lb > 0) & (~np.isnan(nd))
    
    if not np.any(valid_poly):
        return None

    nd_valid = nd[valid_poly]
    lb_valid = lb[valid_poly]

    unique_ids, inverse, counts = np.unique(
        lb_valid, return_inverse=True, return_counts=True
    )
    means = np.bincount(inverse, weights=nd_valid) / counts

    # Validate NDVI range
    if np.any((means < -1.1) | (means > 1.1)):
        logger.warning(f"NDVI values outside expected range [-1, 1]: min={means.min():.3f}, max={means.max():.3f}")

    cm_valid = cm[valid_poly]
    clear_counts = np.bincount(inverse, weights=cm_valid.astype(int))

    string_uids = [int_to_uid.get(int_id) for int_id in unique_ids]
    valid_mask = [u is not None for u in string_uids]
    
    return pd.DataFrame({
        config.POLY_UID: [u for u in string_uids if u is not None],
        "ndvi": means[valid_mask],
        "count": counts[valid_mask],
        "clear_pixels": clear_counts[valid_mask],
    })


# =========================
# MACRO-REGION PROCESSING
# =========================


def process_tile_from_macro(tile, ndvi_med, clear_mask_med, tile_masks):
    """Worker-level processing: loads mapping from local cache."""
    # Retrieve the mapping from the local process memory
    int_to_uid = get_mapping_on_worker()
    
    """Process tile via spatial slicing from macro-region data."""
    tile_id = tile["tile_id"]
    cache_file = (
        config.OUTPUT_DIR
        / "cache"
        / f"{ndvi_med.attrs['year']}_{ndvi_med.attrs['month']:02d}"
        / f"tile_{tile_id:05d}.parquet"
    )

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    # Resolve Future if needed (when called via dask.delayed)
    from dask.distributed import Future
    if isinstance(tile_masks, Future):
        tile_masks = tile_masks.result()
    
    tiles_mask = tile_masks.get(tile_id)
    if tiles_mask is None:
        logger.debug(f"Tile {tile_id} has no mask, skipping")
        return None

    tx0, ty0, tx1, ty1 = tile["bounds"]

    try:
        tile_ndvi = ndvi_med.sel(x=slice(tx0, tx1), y=slice(ty1, ty0))
        tile_clear = clear_mask_med.sel(x=slice(tx0, tx1), y=slice(ty1, ty0))
    except (KeyError, ValueError):
        logger.debug(f"Tile {tile_id} has no data, skipping")
        return None

    if tile_ndvi.sizes.get("x", 0) == 0 or tile_ndvi.sizes.get("y", 0) == 0:
        logger.debug(
            f"Tile {tile_id} has no data (x or y dimension zero length), skipping"
        )
        return None

    if tile_ndvi.isnull().all():
        logger.debug(f"Tile {tile_id} fully masked, skipping")
        return None

    result = zonal_mean(tile_ndvi, tile_clear, tiles_mask, int_to_uid)
    del tile_ndvi, tile_clear, tiles_mask

    if result is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Failed to cache tile {tile_id}: {e}")
            cache_file.unlink(missing_ok=True)

    return result


def process_macro_tile(macro_tile, year, month, items, wofs_items):
    """
    Process a macro-tile (8192x8192) by loading Landsat/NDVI once, masking with WOfS,
    aggregating to monthly median, and processing all sub-tiles (2048x2048) sequentially.
    Cached tile results are read if available.
    """

    macro_id = macro_tile["macro_id"]
    tiles = macro_tile["tiles"]
    bbox_wgs84 = macro_tile["bounds_wgs84"]

    logger.info(f"   Macro-tile {macro_id}: {len(tiles)} sub-tiles")

    cache_dir = config.OUTPUT_DIR / "cache" / f"{year}_{month:02d}"
    cached_tiles = []
    uncached_tiles = []
    
    for tile in tiles:
        cache_file = cache_dir / f"tile_{tile['tile_id']:05d}.parquet"
        if cache_file.exists():
            try:
                cached_tiles.append(pd.read_parquet(cache_file))
            except Exception as e:
                logger.warning(f"Failed to read cached tile {tile['tile_id']:05d}, will regenerate: {e}")
                uncached_tiles.append(tile)
        else:
            uncached_tiles.append(tile)
    
    if not uncached_tiles:
        logger.info(f"   Macro-tile {macro_id} fully cached")
        return cached_tiles

    tile_masks = {tile["tile_id"]: load_tile_mask(tile["tile_id"]) for tile in uncached_tiles}
    tile_masks = {k: v for k, v in tile_masks.items() if v is not None}
    
    dask_client = get_client()
    tile_masks_future = dask_client.scatter(tile_masks, broadcast=True)
    logger.debug(f"   Macro-tile {macro_id}: loading landsat data...")
    try:
        landsat_ds = odc.stac.load(
            items,
            bands=["nbart_red", "nbart_nir", "oa_fmask"],
            bbox=bbox_wgs84,
            crs=config.CRS,
            resolution=30,
            groupby="solar_day",
            chunks={"time": 1, "x": config.TILE_PIXELS, "y": config.TILE_PIXELS},
            fail_on_error=False,
        )
    except Exception as e:
        logger.error(f"Landsat load failed for macro-tile {macro_id}: {e}")
        return cached_tiles
    
    if (
        landsat_ds is None
        or landsat_ds.time.size == 0
        or landsat_ds.sizes["x"] == 0
        or landsat_ds.sizes["y"] == 0
    ):
        logger.warning(f"      Landsat: no data for macro-tile {macro_id} ({year}-{month:02d})")
        return cached_tiles
    else:
        logger.info(f"   Loaded {landsat_ds.time.size}/{len(items)} Landsat scenes for macro-tile {macro_id}")

    landsat_ds = landsat_ds.persist()
    check_dask_graph(landsat_ds, "landsat_ds after persist")
    logger.debug(f"      Memory after Landsat load: {log_memory()}")

    # -------------------------
    # Clear mask & NDVI
    # -------------------------
    clear_mask = ((landsat_ds.oa_fmask == 1) | (landsat_ds.oa_fmask == 5)).persist()

    if clear_mask.isnull().all():
        logger.info(f"      Macro-tile {macro_id}: no clear pixels ({year}-{month:02d})")
        return cached_tiles

    landsat_ds["nbart_red"] = landsat_ds.nbart_red.where(clear_mask)
    landsat_ds["nbart_nir"] = landsat_ds.nbart_nir.where(clear_mask)

    ndvi = compute_ndvi(landsat_ds.nbart_red, landsat_ds.nbart_nir).persist()
    del landsat_ds
    logger.debug(f"      Memory after NDVI compute: {log_memory()}")

    # -------------------------
    # Apply WOfS if available
    # -------------------------
    logger.debug(f"   Macro-tile {macro_id}: loading WOfS data and open water mask")
    if wofs_items:
        try:
            wofs_data = odc.stac.load(
                wofs_items,
                bbox=bbox_wgs84,
                crs=config.CRS,
                resolution=30,
                groupby="solar_day",
                chunks={"time": 1, "x": config.TILE_PIXELS, "y": config.TILE_PIXELS},
                fail_on_error=False,  # stop on failures to diagnose
            )
            if (
                wofs_data is not None
                and "water" in wofs_data
                and wofs_data.time.size > 0
            ):
                logger.info(f"   Loaded {wofs_data.time.size}/{len(wofs_items)} WOfS scenes for macro-tile {macro_id}")
                wofs_data = wofs_data.persist()
                water_int = wofs_data.water.fillna(0).astype("uint8")
                quality_mask = (water_int & 0b01100011) == 0
                open_water = (water_int & (1 << 7)) > 0
                vegetation_mask = ~open_water & quality_mask
                ndvi = ndvi.where(vegetation_mask)
                masked_pixels = vegetation_mask.sum().compute()
                logger.debug(
                    f"      WOfS applied, {masked_pixels:.0f} open water pixels masked"
                )
                del wofs_data, vegetation_mask, water_int
        except Exception as e:
            logger.warning(f"WOfS load or masking failed: {e}", exc_info=True)
    else:
        logger.debug("WOfS data unavailable or empty")

    if ndvi.isnull().all():
        logger.info(f"      Macro-tile {macro_id}: NDVI fully masked ({year}-{month:02d})")
        return cached_tiles

    # -------------------------
    # Monthly median & clear mask aggregation
    # -------------------------
    ndvi_med = ndvi.median("time").where(lambda x: x > 0.0)
    ndvi_med.attrs["year"] = year
    ndvi_med.attrs["month"] = month
    ndvi_med = ndvi_med.persist()
    del ndvi
    check_dask_graph(ndvi_med, "ndvi_med after persist")

    clear_mask_med = clear_mask.max("time")
    clear_mask_med = clear_mask_med.persist()

    # -------------------------
    # Cleanup macro-tile-level Dask objects
    # -------------------------

    del clear_mask

    tile_tasks = []
    for tile in uncached_tiles:
        task = dask.delayed(process_tile_from_macro)(
            tile, ndvi_med, clear_mask_med, tile_masks_future,
        )
        tile_tasks.append(task)

    logger.info(f"   Macro-tile {macro_id}: parallel compute for {len(tile_tasks)} sub-tiles")
    new_results = dask.compute(*tile_tasks)

    del ndvi_med, clear_mask_med
    return cached_tiles + [r for r in new_results if r is not None]


def process_month(year, month, raster_tiles, macro_tiles, dask_client, catalog):
    """Process month using macro-region strategy."""
    final_out = config.OUTPUT_DIR / f"NDVI_{year}_{month:02d}.parquet"
    if final_out.exists():
        logger.info(f"  Found saved result {year}-{month:02d} in {final_out.name} - skipping processing")
        return
    logger.info(f"Processing {year}-{month:02d}")

    all_bounds = [t["bounds"] for t in raster_tiles]
    full_bbox = [
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    ]
    bbox_wgs84 = bbox_to_wgs84(full_bbox)

    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"


    
    logger.info(f"  Searching for Landsat data {start} to {end}...")
    try:
        items = catalog.search(
            collections=config.LANDSAT_COLLECTIONS,
            bbox=bbox_wgs84,
            datetime=f"{start}/{end}",
            limit=300,
        ).item_collection()
    except Exception as e:
        logger.error(f"  Landsat STAC search failed: {e}")
        return
    if len(items) == 0:
        logger.info(f"  No Landsat data for {year}-{month:02d}")
        return
    logger.info(f"  - Found {len(items)} Landsat items")
    
    
    logger.info(f"  Searching for WOfS data to mask open water {start} to {end}...")
    try: 
        wofs_items = catalog.search(
            collections=[config.WOFS_COLLECTION],
            bbox=bbox_wgs84,
            datetime=f"{start}/{end}",
            limit=300,
        ).item_collection()
    except Exception as e:
        logger.error(f"  WOfS STAC search failed: {e}")
        return
    if len(wofs_items) == 0:
        logger.info(f"  No WOfS data for {year}-{month:02d}")
        return
    logger.info(f"  - Found {len(wofs_items)} WOfS items")

    # Gathers results one macro-region at a time
    all_results = []
    for macro_tile in macro_tiles:
        logger.info(f"Macro-tile {macro_tile['macro_id']} - {year}-{month:02d} - memory use: {log_memory()}")
        
        # This call now runs on the main thread, but triggers 
        # parallel work on the workers via dask.compute() inside
        res = process_macro_tile(
            macro_tile, year, month, items, wofs_items,
        )
        
        if res:
            all_results.extend(res)
        
        # FORCE CLEANUP: Tell all workers to clear memory before next macro-tile
        dask_client.run(gc.collect)

    if not all_results:
        logger.info(f"  No results for {year}-{month:02d}. Creating empty baseline.")
        combined = pd.DataFrame(columns=[config.POLY_UID, "ndvi", "count", "clear_pixels", "w_ndvi"])
    else:
        combined = pd.concat(all_results, ignore_index=True)
        combined["w_ndvi"] = combined["ndvi"] * combined["count"]

    aggregated = (
        combined.groupby(config.POLY_UID)
        .agg(
            w_ndvi_sum=("w_ndvi", "sum"),
            count_sum=("count", "sum"),
            clear_pixels_sum=("clear_pixels", "sum"),
        )
        .reset_index()
    )

    mapping_file = config.OUTPUT_DIR / "uid_mapping.json"
    with open(mapping_file) as f:
        data = json.load(f)
    all_uids_df = pd.DataFrame(data["polygons"], columns=["int_value", config.POLY_UID, "total_pixels"])
    all_uids_df = all_uids_df[[config.POLY_UID, "total_pixels"]]  # Keep only UID and pixel count

    final_result = all_uids_df.merge(aggregated, on=config.POLY_UID, how="left")

    final_result["w_ndvi_sum"] = final_result["w_ndvi_sum"].fillna(0)
    final_result["count_sum"] = final_result["count_sum"].fillna(0)
    final_result["clear_pixels_sum"] = final_result["clear_pixels_sum"].fillna(0)

    final_result["ndvi"] = np.where(
        final_result["count_sum"] > 0,
        final_result["w_ndvi_sum"] / final_result["count_sum"],
        np.nan
    )

    final_result["quality"] = (
        final_result["clear_pixels_sum"] / final_result["total_pixels"]
    ).fillna(0)

    final_result["year"] = year
    final_result["month"] = month
    
    final_result = final_result.rename(columns={
        "count_sum": "count",
        "clear_pixels_sum": "clear_pixels"
    })
    
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
    setup_logging(config.LOG_DIR, "ndvi_processing")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    raster_tiles = load_or_create_raster_tiles(config.POLYGON_PATH)
    macro_tiles = create_macro_tiles(raster_tiles)
    
    if config.END_DATE is None:
        current_date = datetime.now()
        end_year, end_month = current_date.year, current_date.month - 1
        if end_month == 0:
            end_year -= 1
            end_month = 12
    else:
        end_year, end_month = config.END_DATE
    start_year, start_month = config.START_DATE
        
    logger.info(f"Processing {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    

    #persistent session
    catalog = stac_client()
    dask_client = start_dask()
    try:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if year == start_year and month < start_month:
                    continue
                if (year, month) > (end_year, end_month):
                    break
                process_month(year, month, raster_tiles, macro_tiles, dask_client, catalog)
    finally:
        client.close()


if __name__ == "__main__":
    main()
