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
import traceback

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
#from dask.distributed import Client as DaskClient, LocalCluster
from config import ndvi_landsat_cfg as config
from tools.logging_setup import setup_logging
from tools.dask import start_dask

logger = logging.getLogger(__name__)


# =========================
# SPATIAL GRID
# =========================


def get_global_uid_mapping(polygons):
    mapping_file = config.OUTPUT_DIR / "uid_mapping.json"
    if mapping_file.exists():
        with open(mapping_file) as f:
            data = json.load(f)
            logger.debug(f"Loaded UID mapping from {mapping_file}")
            return data["uid_to_int"], data["int_to_uid"]

    unique_uids = sorted(polygons[config.POLY_UID].unique())
    uid_to_int = {uid: i + 1 for i, uid in enumerate(unique_uids)}
    int_to_uid = {str(i + 1): uid for uid, i in uid_to_int.items()}

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(mapping_file, "w") as f:
        json.dump({"uid_to_int": uid_to_int, "int_to_uid": int_to_uid}, f)
        logger.debug(f"Saved UID mapping to {mapping_file}")

    return uid_to_int, int_to_uid


def rasterize_tile_polygons(tile_id, tile_bounds, polygons, uid_to_int, total_pixels):
    minx, miny, maxx, maxy = tile_bounds
    tile_geom = box(minx, miny, maxx, maxy)
    polys = polygons[polygons.intersects(tile_geom)]

    if polys.empty:
        return

    tiles_file = config.OUTPUT_DIR / "tile_masks" / f"tile_{tile_id:05d}.tif"
    tiles_file.parent.mkdir(parents=True, exist_ok=True)

    transform = from_bounds(minx, miny, maxx, maxy, config.TILE_PIXELS, config.TILE_PIXELS)
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
    for uid_int, count in zip(unique, counts):
        uid = polys_mapped[polys_mapped["raster_id"] == uid_int][config.POLY_UID].iloc[0]
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
        try:
            dst.write(tiles, 1)
        except Exception as e:
            logger.error(f"Error writing tile {tile_id}: {e}")
            logger.error(traceback.format_exc())
            raise e


def load_or_create_raster_tiles(polygons_path):
    tiles_dir = config.OUTPUT_DIR / "tile_masks"
    pixel_count = config.OUTPUT_DIR / "total_pixels_per_polygon.parquet"
    
    if tiles_dir.exists() and pixel_count.exists():
        return sorted(
            [
                {
                    "tile_id": int(f.stem.split("_")[1]),
                    "bounds": (b.left, b.bottom, b.right, b.top)
                }
                for f in tiles_dir.glob("tile_*.tif")
                for b in [rasterio.open(f).bounds]
            ],
            key=lambda x: x["tile_id"]
        )

    logger.info("Generating raster tiles...")
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True)
    
    polygons = gpd.read_file(polygons_path).to_crs(config.CRS)
    uid_to_int, _ = get_global_uid_mapping(polygons)

    minx, miny, maxx, maxy = polygons.total_bounds
    tile_size = config.TILE_PIXELS * config.PIXEL_SIZE
    xs = np.arange(np.floor(minx / config.PIXEL_SIZE) * config.PIXEL_SIZE, np.ceil(maxx / config.PIXEL_SIZE) * config.PIXEL_SIZE, tile_size)
    ys = np.arange(np.floor(miny / config.PIXEL_SIZE) * config.PIXEL_SIZE, np.ceil(maxy / config.PIXEL_SIZE) * config.PIXEL_SIZE, tile_size)

    total_pixels = {}
    tile_coords = [(x, y) for x in xs for y in ys]
    for tile_id, (x, y) in enumerate(tqdm(tile_coords, desc="Rasterising tiles")):
        rasterize_tile_polygons(tile_id, (x, y, x + tile_size, y + tile_size), polygons, uid_to_int, total_pixels)
    logger.info("Raster tile generation complete.")

    pd.DataFrame(list(total_pixels.items()), columns=[config.POLY_UID, "total_pixels"]).to_parquet(pixel_count)
    logger.info(f"Total pixel count per polygon saved to {pixel_count}.")

    return load_or_create_raster_tiles(polygons_path)


def create_macro_tiles(raster_tiles):
    """Group tiles into macro-regions (8192×8192 pixels)."""
    if not raster_tiles:
        return []

    macro_size = config.TILE_PIXELS * config.PIXEL_SIZE * config.MACRO_TILE_FACTOR

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
            macro_bbox = (x, y, min(x + macro_size, maxx), min(y + macro_size, maxy))

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

            x += macro_size
        y += macro_size

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
    if any(not np.isfinite(x) for x in bbox):
        raise ValueError(f"Invalid bbox: {bbox}")
    geom = Geometry(geo_box(*bbox, GeoCRS(config.CRS)))
    geom_wgs84 = geom.to_crs("EPSG:4326").boundingbox
    return [geom_wgs84.left, geom_wgs84.bottom, geom_wgs84.right, geom_wgs84.top]


def stac_client():
    return Client.open(config.STAC_URL)


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
    """Calculate zonal statistics with NaN handling and clear pixel counts."""
    nd = ndvi.values
    cm = clear_mask.values
    lb = tiles

    valid_poly = lb > 0
    if not valid_poly.any():
        return None

    valid_ndvi = (~np.isnan(nd)) & valid_poly
    nd_valid = nd[valid_ndvi]
    lb_ndvi = lb[valid_ndvi]

    unique_ids, inverse, counts = np.unique(lb_ndvi, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=nd_valid) / counts

    lb_all = lb[valid_poly]
    cm_all = cm[valid_poly]
    unique_ids_clear, inverse_clear = np.unique(lb_all, return_inverse=True)[:2]
    clear_counts = np.bincount(inverse_clear, weights=np.nan_to_num(cm_all, nan=0).astype(int))

    string_uids = [int_to_uid[str(int_id)] for int_id in unique_ids]
    clear_dict = dict(zip(unique_ids_clear, clear_counts))
    clear_pixel_counts = [clear_dict.get(uid, 0) for uid in unique_ids]

    return pd.DataFrame({config.POLY_UID: string_uids, "ndvi": means, "count": counts, "clear_pixels": clear_pixel_counts})


# =========================
# MACRO-REGION PROCESSING
# =========================


def process_tile_from_macro(tile, ndvi_med, clear_mask_med, tile_masks, int_to_uid):
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
        logger.debug(f"Tile {tile_id} has no data (x or y dimension zero length), skipping")
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


def process_macro_tile(macro_tile, year, month, items, wofs_items, int_to_uid, client):
    """
    Process a macro-tile (8192x8192) by loading Landsat/NDVI once, masking with WOfS,
    aggregating to monthly median, and processing all sub-tiles (2048x2048) sequentially.
    Cached tile results are read if available.
    """
    
    
    macro_id = macro_tile["macro_id"]
    tiles = macro_tile["tiles"]
    bbox_wgs84 = macro_tile["bounds_wgs84"]  # Use precomputed WGS84 bbox
    
    logger.info(f"Macro-tile {macro_id}: {len(tiles)} tiles")
    
    # -------------------------
    # Load or skip cached tiles
    # -------------------------
    cache_dir = config.OUTPUT_DIR / "cache" / f"{year}_{month:02d}"
    all_cached = all(
        (cache_dir / f"tile_{tile['tile_id']:05d}.parquet").exists()
        for tile in tiles
    )
    if all_cached:
        logger.info(f"Macro-tile {macro_id} fully cached, reading from disk")
        results = []
        for tile in tiles:
            path = cache_dir / f"tile_{tile['tile_id']:05d}.parquet"
            try:
                results.append(pd.read_parquet(path))
            except Exception as e:
                logger.error(f"Failed to read cached tile {tile['tile_id']:05d}: {e}")
        return results
    
       
    # -------------------------
    # Load tile masks
    # -------------------------
    tile_masks = {tile["tile_id"]: load_tile_mask(tile["tile_id"]) for tile in tiles}
    tile_masks = {k: v for k, v in tile_masks.items() if v is not None}
    # -------------------------
    # Load Landsat data
    # -------------------------
    try:
        landsat_ds = odc.stac.load(
            items,
            bands=["nbart_red", "nbart_nir", "oa_fmask"],
            bbox=bbox_wgs84,
            crs=config.CRS,
            resolution=30,
            groupby="solar_day",
            chunks={"time": 1, "x": config.TILE_PIXELS, "y": config.TILE_PIXELS},
            fail_on_error=True,  # stop on failures to diagnose
        )
    except Exception as e:
        logger.error(f"Landsat load failed for macro-tile {macro_id}: {e}", exc_info=True)
        return []
          
    if landsat_ds is None or landsat_ds.time.size == 0 or landsat_ds.sizes["x"] == 0 or landsat_ds.sizes["y"] == 0:
        logger.warning(f"      Landsat: no data loaded for macro-tile {macro_id} for {year}-{month:02d}")
        return []
            
    landsat_ds = landsat_ds.persist()
    check_dask_graph(landsat_ds, "landsat_ds after persist")
    logger.debug(f"      Memory after Landsat load: {log_memory()}")
    
    # -------------------------
    # Clear mask & NDVI
    # -------------------------
    clear_mask = ((landsat_ds.oa_fmask == 1) | (landsat_ds.oa_fmask == 5)).persist()
    
    if clear_mask.isnull().all():
        logger.info(
            f"      Macro-tile {macro_id}: no clear pixels ({year}-{month:02d}), skipping"
        )
        return []
    
    landsat_ds["nbart_red"] = landsat_ds.nbart_red.where(clear_mask)
    landsat_ds["nbart_nir"] = landsat_ds.nbart_nir.where(clear_mask)

    ndvi = compute_ndvi(landsat_ds.nbart_red, landsat_ds.nbart_nir).persist()
    logger.debug(f"      Memory after NDVI compute: {log_memory()}")
    
    # -------------------------
    # Apply WOfS if available
    # -------------------------
    if wofs_items:
        try:
            wofs_data = odc.stac.load(
                wofs_items,
                bbox=bbox_wgs84,
                crs=config.CRS,
                resolution=30,
                groupby="solar_day",
                chunks={"time": 1, "x": config.TILE_PIXELS, "y": config.TILE_PIXELS},
                fail_on_error=True, # stop on failures to diagnose
            )
            if wofs_data is not None and "water" in wofs_data and wofs_data.time.size > 0:
                wofs_data = wofs_data.persist()
                water_int = wofs_data.water.fillna(0).astype("uint8")
                quality_mask = (water_int & 0b01100011) == 0
                open_water = (water_int & (1 << 7)) > 0
                vegetation_mask = ~open_water & quality_mask
                ndvi = ndvi.where(vegetation_mask)
                masked_pixels = vegetation_mask.sum().compute()
                logger.debug(f"      WOfS applied, {masked_pixels:.0f} open water pixels masked")
                del wofs_data, vegetation_mask, water_int
        except Exception as e:
            logger.warning(f"WOfS load or masking failed: {e}", exc_info=True)
    else:
        logger.debug("WOfS data unavailable or empty")

    if ndvi.isnull().all():
        logger.info(
            f"      Macro-tile {macro_id}: NDVI fully masked ({year}-{month:02d}), skipping"
        )
        return []

    # -------------------------
    # Monthly median & clear mask aggregation
    # -------------------------
    ndvi_med = ndvi.median("time").where(lambda x: x > 0.0)
    ndvi_med.attrs["year"] = year
    ndvi_med.attrs["month"] = month
    ndvi_med = ndvi_med.persist()
    check_dask_graph(ndvi_med, "ndvi_med after persist")

    clear_mask_med = clear_mask.max("time")
    clear_mask_med = clear_mask_med.persist()

    # -------------------------
    # Cleanup macro-tile-level Dask objects
    # -------------------------

    del landsat_ds, ndvi, clear_mask


   # Process tiles sequentially inside the macro tile, and let Dask parallelise internally:
    results = []
    for tile in tiles:
        try:
            r = process_tile_from_macro(tile, ndvi_med, clear_mask_med, tile_masks, int_to_uid)
            if r is not None:
                results.append(r)
        except Exception as e:
            logger.error(f"      Error processing tile {tile['tile_id']:05d}: {e}", exc_info=True)
            continue
    # Cleanup
    del ndvi_med, clear_mask_med
    client.run(gc.collect)
    logger.debug(f"      Memory after macro-tile cleanup: {log_memory()}")

    return [r for r in results if r is not None]


def process_month(year, month, raster_tiles, macro_tiles, client):
    """Process month using macro-region strategy."""
    final_out = config.OUTPUT_DIR / f"NDVI_{year}_{month:02d}.parquet"
    if final_out.exists():
        logger.info(f"  Skipping {year}-{month:02d}")
        return

    with open(config.OUTPUT_DIR / "uid_mapping.json") as f:
        int_to_uid = json.load(f)["int_to_uid"]

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

    catalog = stac_client()
    try:
        items = catalog.search(
            collections=config.LANDSAT_COLLECTIONS, bbox=bbox_wgs84, datetime=f"{start}/{end}"
        ).item_collection()
    except Exception as e:
        logger.error(f"  Landsat STAC search failed: {e}")
        return
    if len(items) == 0:
        logger.info(f"  No Landsat data for {year}-{month:02d}")
        return
    
    try:
        wofs_items = catalog.search(
            collections=[config.WOFS_COLLECTION], bbox=bbox_wgs84, datetime=f"{start}/{end}"
        ).item_collection()
    except Exception as e:
        logger.error(f"  WOfS STAC search failed: {e}")
        return
    if len(wofs_items) == 0:
        logger.info(f"  No WOfS data for {year}-{month:02d}")
        return

    logger.info(f"  Found {len(items)} Landsat items, {len(wofs_items)} WOfS items")

    all_results = []
    for macro_tile in macro_tiles:
        results = process_macro_tile(
            macro_tile, year, month, items, wofs_items, int_to_uid, client
        )
        all_results.extend(results)

    if not all_results:
        logger.info(f"  No results for {year}-{month:02d}")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined["w_ndvi"] = combined["ndvi"] * combined["count"]

    final_result = (
        combined.groupby(config.POLY_UID)
        .agg(w_ndvi=("w_ndvi", "sum"), count=("count", "sum"), clear_pixels=("clear_pixels", "sum"))
        .reset_index()
    )

    final_result["ndvi"] = final_result["w_ndvi"] / final_result["count"]
    final_result.drop(columns="w_ndvi", inplace=True)

    total_pixels_file = config.OUTPUT_DIR / "total_pixels_per_polygon.parquet"
    if total_pixels_file.exists():
        total_pixels_df = pd.read_parquet(total_pixels_file)
        final_result = final_result.merge(total_pixels_df, on=config.POLY_UID, how="left")
        final_result["quality"] = final_result["clear_pixels"] / final_result["total_pixels"]
    else:
        logger.warning("  total_pixels_per_polygon.parquet not found. Quality calculation skipped.")

    final_result["year"] = year
    final_result["month"] = month
    #atomic write
    try:
        tmp_out = final_out.with_suffix(".tmp.parquet")
        final_result.to_parquet(tmp_out)    
        tmp_out.rename(final_out)
        logger.info(f"  Completed {year}-{month:02d}: {len(final_result)} polygons")
    except Exception as e:
        logger.error(f"  Failed to write final output for {year}-{month:02d}: {e}", exc_info=True)
        tmp_out.unlink(missing_ok=True)


# =========================
# MAIN
# =========================


def main():
    setup_logging(config.LOG_DIR, "ndvi_processing")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = start_dask()
    raster_tiles = load_or_create_raster_tiles(config.POLYGON_PATH)
    macro_tiles = create_macro_tiles(raster_tiles)

    logger.info(f"Tiles: {len(raster_tiles)}, Macro-tiles: {len(macro_tiles)}")

    current_date = datetime.now()
    end_year, end_month = current_date.year, current_date.month - 1
    if end_month == 0:
        end_year -= 1
        end_month = 12

    for year in range(config.START_YEAR, end_year + 1):
        for month in range(1, 13):
            if (year, month) > (end_year, end_month):
                break
            logger.info(f"Processing {year}-{month:02d}")
            process_month(year, month, raster_tiles, macro_tiles, client)

    client.close()


if __name__ == "__main__":
    main()
