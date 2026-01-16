import os
import logging

import psutil
from dask.distributed import Client as DaskClient, LocalCluster
from config import ndvi_landsat_cfg as config

logger = logging.getLogger(__name__)

# =========================
# DASK SETUP
# =========================

def start_dask(workers=None):
    """Start a Dask local cluster with optimized settings."""   
    
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    n_cores = os.cpu_count()
    
    # We leave 20% or 4GB (whichever is larger) for the OS to prevent freezing
    usable_ram = max(total_memory_gb * 0.8, total_memory_gb - 4)
    # Small number of workers for STAC loading efficiency
    n_workers = workers or max(1, n_cores // 8)
    threads_per_worker = max(4, n_cores // n_workers)
    memory_limit_per_worker = int(usable_ram // n_workers)

    cluster = LocalCluster(
        processes=True,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=f"{memory_limit_per_worker}GB",
        dashboard_address=f":{config.DASK_DASHBOARD_PORT}",
        silence_logs=logging.ERROR,
        env={
            # Public cloud buckets (DEA, AWS Open Data, etc.)
            "AWS_NO_SIGN_REQUEST": "YES",

            # Robust HTTP behavior for flaky networks
            "GDAL_HTTP_MAX_RETRY": "10",
            "GDAL_HTTP_RETRY_DELAY": "3",
            "GDAL_HTTP_TIMEOUT": "45",

            # Prevent expensive LIST / directory probes on S3 / HTTP
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",

            # Small in-process read cache (per worker)
            "VSI_CACHE": "TRUE",
            "VSI_CACHE_SIZE": "10485760",  # 10 MB
            
            # Enable HTTP/2 for faster concurrent header requests
            "GDAL_HTTP_VERSION": "2",

            # Important: Keep connections open between internal rasterio/gdal calls
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",

            # Use persistent connections across your workers
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.vrt",
        },
    )

    logger.info(f"Dask cluster: {n_workers} workers")
    logger.info(f"  x {threads_per_worker} threads per worker")
    logger.info(f"  x {memory_limit_per_worker} GB worker memory limit")
    logger.info(f"Dashboard: http://127.0.0.1:{config.DASK_DASHBOARD_PORT}/status")

    return DaskClient(cluster)