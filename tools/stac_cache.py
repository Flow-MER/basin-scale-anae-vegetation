"""
STAC Asset Caching Module
=========================

Provides a thread-safe, persistent local cache for STAC assets (e.g., GeoTIFFs) fetched from remote URLs.

Key Features:
- **Atomic Writes**: Uses temporary files and atomic renames to ensure partial downloads are never read.
- **Rate Limiting**: Global token bucket limiter to prevent saturating network bandwidth.
- **Prioritization**: Downloads can be prioritized (e.g., based on spatial location) to optimize processing pipelines.
- **Resilience**: Handles network retries and falls back to remote URLs if caching fails.
- **Cascading Fallbacks**: Supports multiple fallback caches configured via .env file.
"""
import os
import time
import random
import shutil
import logging
import threading
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from functools import wraps
import pystac
from inspect import signature
import gzip

# Load .env file at module import time
try:
    from dotenv import load_dotenv, find_dotenv
    # Try to find .env in current directory or parents
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        logging.getLogger(__name__).info(f"Loaded .env from: {dotenv_path}")
    else:
        logging.getLogger(__name__).debug("No .env file found")
except ImportError:
    logging.getLogger(__name__).warning("python-dotenv not installed, using environment variables only")

import requests


logger = logging.getLogger(__name__)

# needed for cache_search decorator
_cache_instance = None

def configure_cache(*args, **kwargs):
    global _cache_instance
    _cache_instance = STACCache(*args, **kwargs)

def cache_search(func):
    if _cache_instance is None:
        return func
    return _cache_instance.cache_search(func)

class RateLimiter:
    """
    Thread-safe token bucket for global bandwidth throttling.
    
    Ensures that the aggregate download rate across all threads does not exceed
    the specified limit.
    """
    def __init__(self, max_rate_mb):
        """
        Args:
            max_rate_mb (float): Maximum allowed speed in Megabytes per second.
        """
        self.rate_per_sec = max_rate_mb * 1024 * 1024 if max_rate_mb else 0
        self.tokens = self.rate_per_sec
        self.last_check = time.time()
        self._lock = threading.Lock()

    def consume(self, amount):
        """
        Consumes tokens for the given byte amount. Blocks (sleeps) if insufficient tokens.
        """
        if self.rate_per_sec <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self.last_check
            self.last_check = now
            self.tokens += elapsed * self.rate_per_sec
            if self.tokens > self.rate_per_sec:
                self.tokens = self.rate_per_sec
            self.tokens -= amount
            wait_time = -self.tokens / self.rate_per_sec if self.tokens < 0 else 0
        if wait_time > 0:
            time.sleep(wait_time)


class STACCache:
    """
    Manages local caching of remote STAC assets.
    
    This class handles:
    1.  Mapping remote URLs to local file paths.
    2.  Downloading files in the background using a thread pool.
    3.  Managing disk space and cleaning up stale temporary files.
    4.  Providing a `patch_url` method compatible with `odc.stac.load`.
    
    The cache is persistent across runs if the `cache_root` remains the same.
    
    Configuration via .env file:
    ---------------------------
    STAC_CACHE_ROOT       - Primary cache location (required if using get_instance())
    STAC_CACHE_FALLBACKS  - Comma-separated list of fallback caches
    STAC_CACHE_COPY_UP    - Set to "1" to enable copy-up from fallbacks
    STAC_CACHE_MAX_WORKERS - Number of download worker threads
    PATCH_URL_DELAY       - Delay in seconds for patch_url wait
    STAC_CACHE_MAX_RATE_MB - Rate limit in MB/s
    
    Example .env:
    ------------
    STAC_CACHE_ROOT=e:/dea-local-cache
    STAC_CACHE_FALLBACKS=r:/dea-master,t:/dea-archive
    STAC_CACHE_COPY_UP=0
    STAC_CACHE_MAX_WORKERS=8
    PATCH_URL_DELAY=3
    """
    _instance = None
    _instance_lock = threading.Lock()
    
    def __init__(
        self,
        cache_root,
        fallback_cache=None,
        copy_up=False,
        min_free_space_mb=1000,
        max_workers=None,
        read_timeout=600,
        max_rate_mb=None,
        url_list_capacity=10000,
        patch_url_delay=0.1,
        enabled=True,
        readonly=False,
        cache_searches=False,
    ):
        """
        Args:
            cache_root (str): Local directory to store cached files (required)
            fallback_cache (STACCache): Another cache instance to check on miss
            copy_up (bool): If True, copy files from fallback to this cache
            min_free_space_mb (int): Minimum free disk space (MB) required
            max_workers (int): Number of download threads
            read_timeout (int): Timeout in seconds for read operations
            max_rate_mb (float): Global bandwidth limit in MB/s
            url_list_capacity (int): Max number of URL mappings to keep in memory
            patch_url_delay (float): Seconds to wait for in-flight downloads
            enabled (bool): Enable this cache
            readonly (bool): Read-only mode (no downloads)
        """
        if not enabled:
            self.enabled = False
            self._executor = None
            logger.info("STACCache disabled")
            return

        if not cache_root:
            raise ValueError("cache_root is required")

        self.cache_root = Path(cache_root)
        self.enabled = True
        self.readonly = bool(readonly)
        self.fallback_cache = fallback_cache
        self.copy_up = bool(copy_up)
        self.cache_searches = bool(cache_searches)

        try:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            if not self.readonly and not os.access(self.cache_root, os.W_OK):
                logger.warning(f"No write access to {self.cache_root}, switching to read-only")
                self.readonly = True
                
            if min_free_space_mb < 0:
                logger.warning(f"Invalid min_free_space_mb={min_free_space_mb}, defaulting to 1000")
                min_free_space_mb = 1000
            self.min_free_space = min_free_space_mb * 1024 * 1024
            
            if not self.readonly:
                self.enabled = self._check_disk_space(self.min_free_space)
                if not self.enabled:
                    logger.warning(f"Insufficient disk space, cache disabled")

            self._url_map = OrderedDict()
            self._inflight = {}
            self._lock = threading.Lock()
            
            self._limiter = RateLimiter(max_rate_mb)
            
            if patch_url_delay < 0 or patch_url_delay >= 5:
                logger.warning(f"Invalid patch_url_delay={patch_url_delay}, must be >=0 and <5, defaulting to 0.1")
                patch_url_delay = 0.1
            self.patch_url_delay = float(patch_url_delay)
          
            if url_list_capacity <= 0:
                logger.warning(f"Invalid url_list_capacity={url_list_capacity}, defaulting to 10000")
                url_list_capacity = 10000
            self._url_list_capacity = int(url_list_capacity)

            cpu_count = os.cpu_count() or 4
            
            if read_timeout < 0:
                logger.warning(f"Invalid read_timeout={read_timeout}, defaulting to 600")
                read_timeout = 600
            self._timeout = (30, read_timeout)
            
            if max_workers is None:
                max_workers = min(16, max(4, cpu_count))
            self.workers = int(max_workers)
            
            if not self.readonly:
                self._executor = ThreadPoolExecutor(max_workers=self.workers)
            else:
                self._executor = None

            # STAC search cache directory
            self.search_cache_dir = self.cache_root / "stac-search-cache"
            if not self.readonly:
                self.search_cache_dir.mkdir(exist_ok=True)

            logger.info(
                f"STACCache initialized: {self.cache_root} "
                f"(workers={self.workers if not self.readonly else 0}, "
                f"readonly={self.readonly}, "
                f"copy_up={self.copy_up}, "
                f"fallback={'yes' if fallback_cache else 'no'})"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize cache at {cache_root}: {e}")
            self.enabled = False
            self._executor = None
            raise
            
    @classmethod
    def get_instance(cls, **kwargs):
        """
        Singleton accessor with configuration from .env file.
        
        Configuration is read from environment variables (typically loaded from .env):
        
        Required:
            STAC_CACHE_ROOT - Primary cache location
        
        Optional:
            STAC_CACHE_FALLBACKS - Comma-separated list of fallback cache paths
            STAC_CACHE_COPY_UP - Set to "1" to enable copy-up
            STAC_CACHE_MAX_WORKERS - Number of worker threads
            PATCH_URL_DELAY - Delay for patch_url (seconds)
            STAC_CACHE_MAX_RATE_MB - Rate limit in MB/s
            STAC_CACHE_READONLY - Set to "1" for read-only mode
            STAC_CACHE_SEARCHES - cache pystac catalog search results
        
        Keyword arguments override environment variables.
        
        Example .env:
            STAC_CACHE_ROOT=e:/dea-local-cache
            STAC_CACHE_FALLBACKS=r:/dea-master,t:/dea-archive
            STAC_CACHE_MAX_WORKERS=8
            PATCH_URL_DELAY=3
            
        
        Usage:
            cache = STACCache.get_instance()
            patch_url = cache.patch_url
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    config = cls._build_config_from_env(**kwargs)
                    cls._instance = cls(**config)
        return cls._instance
    
    @classmethod
    def _build_config_from_env(cls, **kwargs):
        """
        Build configuration from environment variables.
        
        Creates a cascading chain of fallback caches from STAC_CACHE_FALLBACKS.
        
        Priority: kwargs > environment variables
        """
        config = kwargs.copy()
        
        # Primary cache root (required)
        if 'cache_root' not in config:
            cache_root = os.getenv('STAC_CACHE_ROOT')
            if not cache_root:
                raise ValueError(
                    "STAC_CACHE_ROOT environment variable is required. "
                    "Set it in your .env file or pass cache_root= parameter."
                )
            config['cache_root'] = cache_root
        
        # Build cascading fallback chain
        if 'fallback_cache' not in config:
            fallback_chain = cls._build_fallback_chain_from_env()
            if fallback_chain:
                config['fallback_cache'] = fallback_chain
        
        # Copy-up setting
        if 'copy_up' not in config:
            copy_up_env = os.getenv('STAC_CACHE_COPY_UP', '').lower()
            config['copy_up'] = copy_up_env in ('1', 'true', 'yes')
        
        # Max workers
        if 'max_workers' not in config:
            max_workers_env = os.getenv('STAC_CACHE_MAX_WORKERS')
            if max_workers_env and max_workers_env.isdigit():
                config['max_workers'] = int(max_workers_env)
        
        # Patch URL delay
        if 'patch_url_delay' not in config:
            delay_env = os.getenv('PATCH_URL_DELAY')
            if delay_env:
                try:
                    config['patch_url_delay'] = float(delay_env)
                except ValueError:
                    logger.warning(f"Invalid PATCH_URL_DELAY: {delay_env}")
        
        # Rate limit
        if 'max_rate_mb' not in config:
            rate_env = os.getenv('STAC_CACHE_MAX_RATE_MB')
            if rate_env:
                try:
                    config['max_rate_mb'] = float(rate_env)
                except ValueError:
                    logger.warning(f"Invalid STAC_CACHE_MAX_RATE_MB: {rate_env}")
        
        # Read-only mode
        if 'readonly' not in config:
            readonly_env = os.getenv('STAC_CACHE_READONLY', '').lower()
            config['readonly'] = readonly_env in ('1', 'true', 'yes')
        
        #Cache catalog search results
        if 'cache_searches' not in config:
            cache_search_env = os.getenv('STAC_CACHE_SEARCHES', '').lower()
            config['cache_searches'] = cache_search_env in ('1', 'true', 'yes')
        
        return config
    
    @classmethod
    def _build_fallback_chain_from_env(cls):
        """
        Build cascading chain of fallback caches from STAC_CACHE_FALLBACKS.
        
        STAC_CACHE_FALLBACKS should be a comma-separated list of paths:
            STAC_CACHE_FALLBACKS=r:/dea-master,t:/dea-archive,//server/shared
        
        This creates a chain: master → archive → shared
        
        Returns:
            Head of the fallback chain (first cache to check), or None
        """
        fallbacks_env = os.getenv('STAC_CACHE_FALLBACKS')
        if not fallbacks_env:
            return None
        
        # Split by comma and clean whitespace
        fallback_paths = [p.strip() for p in fallbacks_env.split(',') if p.strip()]
        
        if not fallback_paths:
            return None
        
        logger.info(f"Building fallback chain from: {fallback_paths}")
        
        # Build chain in reverse order (last → first)
        # So the first path in the list becomes the head of the chain
        chain = None
        
        for path_str in reversed(fallback_paths):
            path = Path(path_str)
            
            if not path.exists():
                logger.warning(f"Fallback cache path does not exist: {path_str}")
                continue
            
            if cls._is_unc_path(path):
                logger.info(f"Network path: {path_str}")
                # Don't hang if network is down
                if not cls._validate_path_with_timeout(path, timeout=5):
                    logger.warning(f"Network timeout: {path_str}")
            else:
                if not path.exists():
                    continue    
            
            try:
                # Create read-only cache instance
                # Each fallback links to the previous one
                chain = cls(
                    cache_root=str(path),
                    readonly=True,
                    enabled=True,
                    fallback_cache=chain  # Link to previous in chain
                )
                logger.info(f"Added fallback cache: {path_str}")
            except Exception as e:
                logger.error(f"Failed to create fallback cache for {path_str}: {e}")
                continue
        
        if chain:
            # Count the chain length for logging
            depth = 0
            current = chain
            while current:
                depth += 1
                current = current.fallback_cache
            logger.info(f"Fallback chain depth: {depth}")
        
        return chain
    
    @classmethod
    def reset_instance(cls):
        """
        Reset singleton instance.
        
        Useful for testing or reconfiguration.
        """
        with cls._instance_lock:
            if cls._instance:
                try:
                    cls._instance.close()
                except Exception as e:
                    logger.warning(f"Error closing cache instance: {e}")
            cls._instance = None
            
    def __enter__(self):
        """Support 'with STACCache() as cache:' syntax."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure clean shutdown on context exit."""
        self.close()
        return False
    
    def _check_fallback(self, url):
        """
        Check if URL exists in fallback cache chain.
        
        Recursively checks through the entire fallback chain.
        
        Returns:
            Local path if found in any fallback, None otherwise
        """
        if not self.fallback_cache:
            return None
        
        try:
            result = self.fallback_cache.patch_url(url)
            if result != url:  # Found in fallback
                return result
        except Exception as e:
            logger.warning(f"Error checking fallback cache: {e}")
        
        return None
    
    def _copy_from_fallback(self, source_path, url):
        """
        Copy file from fallback cache to this cache.
        
        Args:
            source_path: Path to file in fallback cache
            url: Original URL (for mapping)
            
        Returns:
            Path to copied file, or original source_path if copy fails
        """
        if self.readonly or not self.enabled:
            return source_path
        
        try:
            # Determine destination
            s3_path = self._url_to_path(url)
            dest_path = self.cache_root / Path(s3_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic copy (via temp file)
            tmp_path = dest_path.with_suffix('.tmp')
            shutil.copy2(source_path, tmp_path)
            os.replace(tmp_path, dest_path)
            
            # Update memory map
            with self._lock:
                while len(self._url_map) >= self._url_list_capacity:
                    self._url_map.popitem(last=False)
                self._url_map[url] = str(dest_path)
            
            file_size = Path(dest_path).stat().st_size / (1<<20)
            logger.debug(f"Copied from fallback: {source_path} -> {dest_path} ({file_size:.1f} MB)")
            
            return str(dest_path)
        
        except Exception as e:
            logger.warning(f"Copy-up failed for {url}: {e}")
            return source_path

    def _check_disk_space(self, required_bytes=0):
        """Verifies sufficient disk space is available before starting a download."""
        try:
            _, _, free = shutil.disk_usage(str(self.cache_root))
            return free >= self.min_free_space + required_bytes
        except Exception as e:
            logger.warning(f"Failed to check disk space: {e}")
            return False
    
    @staticmethod
    def _is_unc_path(path):
        return str(path).startswith(('\\\\', '//'))
    
    @staticmethod
    def _validate_path_with_timeout(path, timeout=5):
        result = {'exists': False}
        def check():
            try:
                result['exists'] = Path(path).is_file()
            except:
                result['exists'] = False
        
        thread = threading.Thread(target=check, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        return result['exists']
    
    @staticmethod
    def _url_to_path(url):
        """
        Normalise any DEA URL form to a local path structure including the bucket name.
        e.g. s3://dea-public-data/baseline/... -> dea-public-data/baseline/...
        """
        if "data.dea.ga.gov.au" in url:
            return url.replace("https://data.dea.ga.gov.au/", "dea-public-data/")

        if url.startswith("s3://"):
            return url.replace("s3://", "")

        return url

    def _cleanup_local_stale_tmp(self, dir_path, max_age=86400):
        """
        Opportunistically clean stale .tmp files in a specific directory.

        Why this approach?
        - It is called after a successful download completes for a specific directory.
        - This is more efficient than a global, periodic scan, especially when the
          cache has a deep, sparse directory structure.
        - The I/O cost of scanning a single, small directory is negligible and is
          amortized across successful downloads.
        """
        try:
            now = time.time()
            # os.scandir is faster than Path.glob as it avoids creating Path objects
            with os.scandir(dir_path) as it:
                for entry in it:
                    if entry.name.endswith(".tmp") and entry.is_file():
                        if now - entry.stat().st_mtime > max_age:
                            try:
                                os.unlink(entry.path)
                            except OSError:
                                pass
        except OSError:
            pass

    def _download_asset(self, url, priority=None, max_retries=3):
        """
        Download a single asset via HTTPS with atomic write and retries.
        
        Why Atomic Write?
        We download to a .tmp file and then `os.replace` (rename) it to the final filename.
        This ensures that other processes or threads never see a partially downloaded file.
        If the process crashes mid-download, only a .tmp file is left, which is cleaned up later.
        """
        try:
            s3_path = self._url_to_path(url)
            local_dest = self.cache_root / Path(s3_path)
            local_dest.parent.mkdir(parents=True, exist_ok=True)

            # Already cached?
            if local_dest.exists() and local_dest.stat().st_size > 0:
                logger.debug(f"Cached: {s3_path}")
                return url, str(local_dest.resolve())

            # Disk space check
            if not self._check_disk_space():
                logger.warning(f"Skipping cache, low disk space for {s3_path}")
                self.enabled = False
                return url, url

            tmp_dest = local_dest.with_suffix(f".{random.randint(100000, 999999)}.tmp")

            # Construct HTTPS URL
            if s3_path.startswith("dea-public-data/"):
                https_url = f"https://data.dea.ga.gov.au/{s3_path.replace('dea-public-data/', '', 1)}"
            elif url.startswith("http"):
                https_url = url
            else:
                https_url = f"https://data.dea.ga.gov.au/{s3_path}"

            for attempt in range(max_retries):
                try:
                    p_str = f" [p={priority}]" if priority is not None else ""
                    logger.debug(
                        f"Downloading{p_str}: {s3_path} (Attempt {attempt + 1}) "
                        f"[thread={threading.current_thread().name}]"
                    )
                    t0 = time.time()

                    # 10s connect timeout, 300s read timeout
                    # Read timeout is generous: 24 threads sharing ~100 Mbps means
                    # each thread gets ~4 Mbps — a 50 MB file can take 100s+ between chunks.
                    chunk_sz = 1024 * 1024 if self._limiter.rate_per_sec > 0 else 8 * 1024 * 1024
                    with requests.get(https_url, stream=True, timeout=self._timeout) as resp:
                        resp.raise_for_status()
                        with open(tmp_dest, "wb") as local_f:
                            for chunk in resp.iter_content(chunk_size=chunk_sz):
                                if not chunk:
                                    continue

                                self._limiter.consume(len(chunk))
                                local_f.write(chunk)

                    # Atomic rename — no lock needed: tmp name is unique per thread,
                    # and replace() is atomic on NTFS.
                    tmp_dest.replace(local_dest)
                    # Self-cleaning: remove stale .tmp files in this specific directory
                    self._cleanup_local_stale_tmp(local_dest.parent)

                    elapsed = time.time() - t0
                    size_mb = local_dest.stat().st_size / (1024 * 1024)
                    logger.debug(
                        f"Finished{p_str}: {s3_path} in {elapsed:.2f}s ({size_mb:.1f} MB) "
                        f"[thread={threading.current_thread().name}]"
                    )
                    return url, str(local_dest.resolve())

                except Exception as e:
                    logger.warning(f"Error downloading {s3_path} (attempt {attempt + 1}): {e}")
                    if tmp_dest.exists():
                        try:
                            tmp_dest.unlink()
                        except OSError:
                            pass
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)

            logger.error(f"Failed to download {s3_path} after {max_retries} attempts")
            return url, url
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return url, url

    def _get_intersection_priority(self, item_bbox, filter_bboxes, padding=0.2):
        """
        Calculates download priority based on spatial intersection.
        
        Why?
        Processing often happens sequentially (e.g., Macro-tile 1, then 2).
        We want to prioritize downloading assets for Macro-tile 1 so they are ready
        when the processor needs them, rather than downloading random tiles.
        
        Returns the index of the *first* matching bbox in `filter_bboxes`.
        This ensures that if an item intersects multiple bboxes, it is assigned the
        priority of the lowest-index bbox (e.g. earliest macro-tile).

        Args:
            padding (float): Padding in degrees to apply to filter bboxes.
                             0.2 deg is approx km, providing buffer for edge effects.

        Returns None if no intersection.
        """
        if not filter_bboxes:
            return 0

        if not item_bbox or len(item_bbox) != 4:
            return None

        ix0, iy0, ix1, iy1 = item_bbox
        for i, (fx0, fy0, fx1, fy1) in enumerate(filter_bboxes):
            # Check for overlap: not (Left > Right or Right < Left or Top < Bottom or Bottom > Top)
            if (
                ix0 < fx1 + padding
                and ix1 > fx0 - padding
                and iy0 < fy1 + padding
                and iy1 > fy0 - padding
            ):
                return i
        return None

    def _prune_url_map(self, num_to_remove=1000):
        """
        Enforces LRU (Least Recently Used) capacity on the internal URL map.
        
        If _url_map is over capacity, removes the oldest items (Python 3.7+ dicts preserve insertion order).
        """
        # This method should be called within a lock.
        if len(self._url_map) >= self._url_list_capacity:
            # In Python 3.7+, dicts are insertion-ordered. list(keys()) gives us oldest first.
            # We collect keys to remove first to avoid modifying the dict during iteration.
            keys_to_remove = list(self._url_map.keys())[:num_to_remove]
            for key in keys_to_remove:
                self._url_map.pop(key, None)
            logger.debug(
                f"Pruned {len(keys_to_remove)} oldest entries from url_map "
                f"(new size: {len(self._url_map)})"
            )

    def _on_download_done(self, url, fut):
        try:
            _, path = fut.result()
        except Exception:
            path = url
        with self._lock:
            # Simple FIFO eviction (fast, no I/O)
            while len(self._url_map) >= self._url_list_capacity:
                self._url_map.popitem(last=False)  # Remove oldest entry
            
            # Add new entry
            self._url_map[url] = str(path)
            
            # Move to end to mark as recently used (optional, for true LRU)
            self._url_map.move_to_end(url)
            
            # Remove from inflight tracking
            if url in self._inflight:
                del self._inflight[url]
        logger.debug(f"Download completed: {url} -> {path}")
    
    def _safe_callback(self, url, future):
        """Wrapper to catch exceptions in download callbacks."""
        try:
            self._on_download_done(url, future)
        except Exception as e:
            logger.error(f"Callback error for {url}: {e}", exc_info=True)
            # Remove from inflight on error
            with self._lock:
                self._inflight.pop(url, None)

    def _gather_assets(self, items, bands, intersection_filter, seen):
        assets = []
        total_bytes = 0
        
        if not items:
            return assets, total_bytes

        try:
            iterator = iter(items)
        except TypeError:
            logger.warning(f"Items argument is not iterable: {type(items)}")
            return assets, total_bytes

        for item in iterator:
            if not hasattr(item, "assets"):
                continue

            priority = 0
            if intersection_filter:
                if not hasattr(item, "bbox"):
                    continue
                priority = self._get_intersection_priority(item.bbox, intersection_filter)
                if priority is None:
                    continue

            asset_keys = bands if bands is not None else list(item.assets.keys())
            for band in asset_keys:
                if band in item.assets:
                    url = item.assets[band].href
                    if url not in seen:
                        seen.add(url)
                        assets.append((priority, url))
                        size = item.assets[band].extra_fields.get("file:size")
                        if size:
                            total_bytes += size
        return assets, total_bytes

    def _submit_assets(self, assets_to_download, total_expected_bytes):
        if not assets_to_download:
            return

        if total_expected_bytes and not self._check_disk_space(total_expected_bytes):
            logger.warning(
                f"Not enough disk space for {len(assets_to_download)} assets "
                f"(~{total_expected_bytes / (1024 * 1024):.1f} MB)"
            )
            return

        # Sort by priority so early macro-tile assets download first
        assets_to_download.sort(key=lambda x: x[0])
        
        total_expected = f" (~{total_expected_bytes / (1024 * 1024):.1f} MB expected)" if total_expected_bytes > 0 else ""

        logger.info(
            f"Starting {len(assets_to_download)} downloads with {self.workers} workers{total_expected}"
        )
        for i in assets_to_download:
            logger.debug(f"Priority: {i[0]}, URL: {i[1]}")

        for priority, url in assets_to_download:
            with self._lock:
                # Skip already cached
                if url in self._url_map:
                    continue
                # Skip if already queued or running
                if url in self._inflight:
                    continue
                # Submit download with priority metadata
                fut = self._executor.submit(self._download_asset, url, priority=priority)
                self._inflight[url] = (
                    fut,
                    priority,
                )  # store tuple instead of just Future
                fut.add_done_callback(lambda f, u=url: self._safe_callback(u, f))

    def submit_cache_items(self, items, bands=None, intersection_filter=None):
        """
        Pre-cache STAC item assets in parallel.

        Args:
            items: Iterable of pystac.Item objects.
            bands: List of band/asset keys to cache. If None, caches all assets.
            intersection_filter: List of [minx, miny, maxx, maxy] bboxes. If provided, only items intersecting at least one bbox are cached.
        """
        # --- Guard clauses for pre-caching ---
        # Why: Exit early if the cache is not in a state to perform write operations.
        if not self.enabled:
            logger.debug("Cache is disabled; skipping pre-cache.")
            return
        if self.readonly:
            logger.debug("Cache is in read-only mode; skipping pre-cache.")
            return

        # Why: Validate that `items` is a non-empty list or tuple before proceeding.
        if not hasattr(items, "__iter__"):
            logger.warning(f"Invalid 'items' type for caching. Expected iterable, got {type(items)}.")
            return
        if not items:
            logger.debug("No items provided to cache.")
            return

        # Why: Validate that `bands` is not an empty list. `None` is a valid input
        # which means "all bands", so we should not exit if `bands` is None.
        if isinstance(bands, (list, tuple)) and not bands:
            logger.debug("Empty 'bands' list provided; nothing to cache.")
            return

        seen = set()
        assets, total_bytes = self._gather_assets(items, bands, intersection_filter, seen)
        self._submit_assets(assets, total_bytes)

    def submit_batches(self, batches, intersection_filter=None):
        """
        Submit multiple batches of items to be cached, sorting them collectively by priority.

        Args:
            batches: List of (items, bands) tuples, or a single [items, bands] list.
            intersection_filter: List of [minx, miny, maxx, maxy] bboxes.
        """
        if not self.enabled:
            logger.debug("STAC cache disabled")
            return
        elif self.readonly:
            logger.debug("STAC cache read-only")
            return
        elif not batches:
            logger.debug("No batches to submit")    
            return
        
        # Normalize input to list of batches if it looks like a single [items, bands] pair
        if isinstance(batches, (list, tuple)) and len(batches) == 2:
            second = batches[1]
            # Check if second element is likely 'bands' (list of strings or None)
            is_bands = second is None or (
                isinstance(second, (list, tuple)) and 
                (len(second) == 0 or isinstance(second[0], str))
            )
            if is_bands:
                batches = [batches]

        if not isinstance(batches, (list, tuple)):
            logger.warning(f"submit_batches expected list or tuple, got {type(batches)}")
            return

        seen = set()
        all_assets = []
        total_bytes = 0

        for i, batch in enumerate(batches):
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                logger.warning(f"Skipping invalid batch at index {i}: expected (items, bands)")
                continue
            
            items, bands = batch
            try:
                assets, bytes_ = self._gather_assets(items, bands, intersection_filter, seen)
                all_assets.extend(assets)
                total_bytes += bytes_
            except Exception as e:
                logger.warning(f"Error gathering assets for batch {i}: {e}")

        self._submit_assets(all_assets, total_bytes)

    def patch_url(self, url, timeout=3):
        """
        Resolves a remote URL to a local file path if cached.
        
        Lookup order:
        1. This cache (memory)
        2. This cache (disk)
        3. Fallback chain (recursive)
        4. Trigger download (if writable)
        5. Return original URL
        
        This method is designed to be passed to `odc.stac.load(..., patch_url=cache.patch_url)`.
        
        Logic:
        1. If cached (memory or disk), return local path.
        2. If downloading (inflight), wait briefly (`timeout`) for it to finish.
        3. If not cached, trigger a background download and return the ORIGINAL url.
           This allows the caller (odc.stac) to proceed with a remote read immediately
           while we cache it for next time.
        """
        if not self.enabled:
            # Even if disabled, check fallback
            fallback_path = self._check_fallback(url)
            return fallback_path if fallback_path else url

        with self._lock:
            # Check memory map first
            path = self._url_map.get(url)
            
            # Cached in memory
            if path and Path(path).is_file():
                logger.debug(f"patch_url HIT in memory: {url} -> {path}")
                return path

            # Check disk cache
            local_path = self.cache_root / self._url_to_path(url)
            if local_path.exists() and local_path.stat().st_size > 0:
                resolved_path = str(local_path.resolve())
                # Prune and add to map
                while len(self._url_map) >= self._url_list_capacity:
                    self._url_map.popitem(last=False)
                self._url_map[url] = resolved_path
                logger.debug(f"patch_url HIT on disk: {url} -> {resolved_path}")
                return resolved_path
        
        # Check fallback chain (outside lock)
        fallback_path = self._check_fallback(url)
        if fallback_path:
            logger.debug(f"Found in fallback: {url} -> {fallback_path}, copy_up={self.copy_up}, readonly={self.readonly}")
            if self.copy_up and not self.readonly:
                # Copy to this cache for faster access next time
                new_path = self._copy_from_fallback(fallback_path, url)
                return new_path
            else:
                # Use directly from fallback
                logger.debug(f"Using fallback directly (copy_up disabled or readonly)")
                with self._lock:
                    self._url_map[url] = fallback_path
                return fallback_path
        
        # Not found anywhere - trigger download if writable
        if self.readonly:
            return url

        with self._lock:
            inflight = self._inflight.get(url)
            
            if not inflight:
                # Not cached, not inflight → submit background download
                fut = self._executor.submit(self._download_asset, url)
                self._inflight[url] = (fut, None)
                fut.add_done_callback(lambda f, u=url: self._safe_callback(u, f))
                logger.debug(f"patch_url MISS and REQUESTED: {url}")
            else:
                logger.debug(f"patch_url already in-flight: {url}")
        
        # Short sleep to allow in-flight download to potentially complete
        if self.patch_url_delay > 0:
            time.sleep(self.patch_url_delay)
            
            with self._lock:
                path = self._url_map.get(url)
                if path and Path(path).is_file():
                    logger.debug(f"patch_url HIT after wait: {url}")
                    return path

        return url

    @staticmethod
    def patch_url_static(cache_root):
        """
        Returns a lightweight, read-only patch_url function.
        
        Usage:
            patcher = STACCache.get_static_patcher(r"r:\\dea-local-cache")
            odc.stac.load(..., patch_url=patcher)
        """
        root = Path(cache_root)

        if not root.exists():
            return lambda url: url

        def patch(url):
            try:
                # Use the class's logic to ensure 1:1 compatibility with the writer
                rel_path = STACCache._url_to_path(url)
                local_path = root / rel_path

                if local_path.exists():
                    return str(local_path.resolve())
            except Exception:
                pass
            return url
        return patch

    def clear_url_map(self):
        """Clears the internal URL map to free memory. Does not delete files on disk."""
        with self._lock:
            self._url_map.clear()

    def cleanup_stale_tmp(self, max_age_seconds=86400):
        """Delete .tmp files older than max_age_seconds (default 24h)."""
        if not self.cache_root.exists():
            return

        count = 0
        now = time.time()
        for root, _, files in os.walk(self.cache_root):
            for f in files:
                if f.endswith(".tmp"):
                    p = Path(root) / f
                    try:
                        if now - p.stat().st_mtime > max_age_seconds:
                            p.unlink()
                            count += 1
                    except OSError:
                        pass
        if count > 0:
            logger.info(f"Cleaned up {count} stale .tmp files")

    def info(self):
        """Diagnostics: disk, cache footprint, file age & size distributions, transient file hygiene, collection breakdown."""
        print(f"\n--- Cache Info: {self.cache_root} ---")

        if not self.cache_root.exists():
            print("Cache directory does not exist.")
            return

        # Disk health
        try:
            total, used, free = shutil.disk_usage(self.cache_root)
            print(
                f"Disk: Total={total / (1 << 40):.1f} TB, "
                f"Used={used / (1 << 40):.1f} TB ({used / total * 100:.0f}%), "
                f"Free={free / (1 << 40):.1f} TB"
            )
        except Exception as e:
            print(f"Disk usage check failed: {e}")

        # Collect files
        all_files = [
            Path(root) / f
            for root, _, files in os.walk(self.cache_root)
            for f in files
        ]
        now = time.time()

        def stat_worker(path):
            try:
                st = path.stat()
                return {
                    "path": path,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "lock": path.name.endswith(".lock"),
                    "tmp": path.name.endswith(".tmp"),
                }
            except OSError:
                return None
            
        if self._executor and not self._executor._shutdown:
            # Use parallel processing
            stats = []
            futures = {self._executor.submit(stat_worker, f): f for f in all_files}
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    stats.append(res)
        else:
            # Serial fallback if executor shut down
            stats = []
            for f in all_files:
                res = stat_worker(f)
                if res:
                    stats.append(res)  
            

        stats = []
        futures = {self._executor.submit(stat_worker, f): f for f in all_files}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                stats.append(res)

        # Accumulators
        age_buckets = {"<7d": 0, "7-30d": 0, "1-6mo": 0, ">6mo": 0}
        transient_counts = {"lock": 0, "tmp": 0}
        transient_ages = []
        collections = {}
        unclassified_files = 0
        total_files = 0
        total_bytes = 0

        for s in stats:
            p = s["path"]
            file_size = s["size"]
            mtime = s["mtime"]

            if s["lock"]:
                transient_counts["lock"] += 1
                continue
            if s["tmp"]:
                transient_counts["tmp"] += 1
                transient_ages.append(now - mtime)
                continue

            age_days = (now - mtime) / 86400
            if age_days < 7:
                age_buckets["<7d"] += 1
            elif age_days < 30:
                age_buckets["7-30d"] += 1
            elif age_days < 180:
                age_buckets["1-6mo"] += 1
            else:
                age_buckets[">6mo"] += 1

            total_files += 1
            total_bytes += file_size

            # Collection classification by path structure
            rel_path = p.relative_to(self.cache_root)
            parts = rel_path.parts
            col_name = "Unclassified"
            if "baseline" in parts:
                try:
                    idx = parts.index("baseline")
                    if idx + 1 < len(parts):
                        col_name = parts[idx + 1]
                except ValueError:
                    pass
            else:
                unclassified_files += 1

            collections[col_name] = collections.get(col_name, 0) + file_size

        # Print summary
        print(f"\nCache: {total_bytes / (1 << 30):.2f} GB, {total_files} files")
        print("File age distribution:", age_buckets)
        print("Collection footprint (GB):")
        for col, sz in sorted(collections.items()):
            print(f"  {col}: {sz / (1 << 30):.2f}")

        oldest_tmp = max(transient_ages) if transient_ages else 0
        oldest_str = (
            f"{int(oldest_tmp // 3600)}h {int((oldest_tmp % 3600) // 60)}m"
            if oldest_tmp
            else "N/A"
        )
        print(
            f"Transient files: Lock={transient_counts['lock']}, "
            f"Temp={transient_counts['tmp']} (oldest {oldest_str})"
        )
        if unclassified_files:
            print(f"Unclassified files: {unclassified_files}")
        print("-----------------------------------")

    def cache_search(self, func):
        """
        Decorator to cache STAC search results.

        Handles:
        - 'bbox': list, tuple, or BoundingBox-like object.
        - 'geom': Shapely Polygon/MultiPolygon or GeoDataFrame.
        - 'datetime_str' OR 'start_date'/'end_date'.
        """

        CACHE_VERSION = "v1"

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled or self.readonly or not self.cache_searches:
                return func(*args, **kwargs)

            try:
                # -----------------------------
                # 1. Bind arguments robustly
                # -----------------------------
                sig = signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                params = bound_args.arguments

                # -----------------------------
                # 2. Collections (required)
                # -----------------------------
                collections = params.get("collections")
                if not collections:
                    return func(*args, **kwargs)

                if isinstance(collections, str):
                    collections = [collections]

                # -----------------------------
                # 3. Extract spatial bounds
                # -----------------------------
                bbox_tuple = None
                raw_bbox = params.get("bbox")
                raw_geom = params.get("geom")

                if raw_bbox:
                    if hasattr(raw_bbox, "bounds"):
                        bbox_tuple = (
                            raw_bbox.left,
                            raw_bbox.bottom,
                            raw_bbox.right,
                            raw_bbox.top,
                        )
                    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                        bbox_tuple = tuple(raw_bbox)

                if not bbox_tuple and raw_geom:
                    try:
                        if hasattr(raw_geom, "bounds"):
                            bbox_tuple = tuple(raw_geom.bounds)
                        elif hasattr(raw_geom, "total_bounds"):
                            bbox_tuple = tuple(raw_geom.total_bounds)
                        elif hasattr(raw_geom, "to_crs"):
                            reprojected = raw_geom.to_crs("EPSG:4326")
                            if hasattr(reprojected, "boundingbox"):
                                bb = reprojected.boundingbox
                                bbox_tuple = (
                                    bb.left,
                                    bb.bottom,
                                    bb.right,
                                    bb.top,
                                )
                    except Exception:
                        pass

                if not bbox_tuple:
                    logger.debug("Missing spatial bounds for STAC cache key.")
                    return func(*args, **kwargs)

                # -----------------------------
                # 4. Normalize datetime
                # -----------------------------
                def _normalize_dt(dt):
                    if dt is None:
                        return None
                    if hasattr(dt, "isoformat"):
                        return dt.isoformat()
                    return str(dt)

                dt_str = params.get("datetime_str")
                start_date = params.get("start_date")
                end_date = params.get("end_date")

                if not dt_str:
                    if start_date is not None and end_date is not None:
                        dt_str = f"{_normalize_dt(start_date)}/{_normalize_dt(end_date)}"
                    elif start_date:
                        dt_str = _normalize_dt(start_date)
                    else:
                        dt_str = "ALL"

                # -----------------------------
                # 5. Build stable cache key
                # -----------------------------
                precision = getattr(self, "bbox_precision", 4)

                key_payload = {
                    "version": CACHE_VERSION,
                    "collections": sorted(collections),
                    "bbox": [round(x, precision) for x in bbox_tuple],
                    "datetime": dt_str,
                }

                raw_key = json.dumps(key_payload, sort_keys=True)
                cache_hash = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
                cache_file = self.search_cache_dir / f"{cache_hash}.json.gz"

                # -----------------------------
                # 6. Attempt cache read
                # -----------------------------
                if cache_file.exists():
                    logger.debug(f"Loading cached STAC search: {cache_file.name}")
                    try:
                        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                            data = json.load(f)
                        return pystac.ItemCollection.from_dict(data)
                    except Exception as e:
                        logger.warning(f"Corrupt cache {cache_file}, deleting: {e}")
                        try:
                            cache_file.unlink(missing_ok=True)
                        except Exception:
                            pass

                # -----------------------------
                # 7. Execute wrapped function
                # -----------------------------
                items = func(*args, **kwargs)

                # -----------------------------
                # 8. Write cache (atomic)
                # -----------------------------
                if (
                    not self.readonly
                    and items
                    and hasattr(items, "__len__")
                    and len(items) > 0
                ):
                    try:
                        tmp_file = cache_file.with_suffix(".tmp")
                        with gzip.open(tmp_file, "wt", encoding="utf-8") as f:
                            json.dump(items.to_dict(), f)
                        tmp_file.replace(cache_file)
                        logger.debug(
                            f"Cached {len(items)} items to {cache_file.name}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write STAC cache: {e}")

                return items

            except Exception as e:
                logger.error(
                    f"Error in search cache decorator: {e}",
                    exc_info=True,
                )
                return func(*args, **kwargs)

        return wrapper



    def close(self):
        """Shut down the thread pool executor."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=True)
            logger.info("STACCache executor shut down")


