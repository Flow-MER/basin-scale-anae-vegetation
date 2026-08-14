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

import gzip
import hashlib
import json
import logging
import math
import os
import random
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from inspect import signature
from pathlib import Path

import pystac
import rasterio
from rasterio.enums import Resampling

# Load .env file at module import time
try:
    from dotenv import find_dotenv, load_dotenv

    # Try to find .env in current directory or parents
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        logging.getLogger(__name__).info(f"Loaded .env from: {dotenv_path}")
    else:
        logging.getLogger(__name__).debug("No .env file found")
except ImportError:
    logging.getLogger(__name__).warning(
        "python-dotenv not installed, using environment variables only"
    )

import requests

logger = logging.getLogger(__name__)

# needed for cache_search decorator
_cache_instance = None


def validate_raster(path):
    try:
        with rasterio.open(path) as src:
            for band in range(1, src.count + 1):
                # Read every native TIFF block.
                for _, window in src.block_windows(band):
                    src.read(band, window=window, masked=False)

                # Also read overviews, since odc may use them.
                for factor in src.overviews(band):
                    height = max(1, math.ceil(src.height / factor))
                    width = max(1, math.ceil(src.width / factor))
                    src.read(
                        band,
                        out_shape=(height, width),
                        resampling=Resampling.nearest,
                    )

        return True, None
    except (rasterio.errors.RasterioError, OSError) as exc:
        return False, str(exc)


def validate_raster_open(path):
    """Perform a lightweight raster structure check without reading pixel blocks."""
    try:
        with rasterio.open(path) as src:
            if src.count < 1 or src.width < 1 or src.height < 1:
                return False, "raster has no readable bands or pixels"
        return True, None
    except (rasterio.errors.RasterioError, OSError) as exc:
        return False, str(exc)


def _checksum_ledger_details(cache_root, cache_path):
    """Map a DEA cache-relative asset path to its collection/tile ledger entry."""
    parts = Path(cache_path).parts
    if len(parts) < 7 or parts[0] != "dea-public-data":
        return None

    collection, path_number, row_number = parts[2:5]
    if not path_number.isdigit() or not row_number.isdigit():
        return None

    ledger_path = (
        Path(cache_root)
        / "checksum-sha1"
        / parts[0]
        / parts[1]
        / collection
        / f"{path_number}_{row_number}.json"
    )
    entry_key = Path(*parts[5:]).as_posix()
    return ledger_path, entry_key, collection, f"{path_number}_{row_number}"


def _read_checksum_ledger(details):
    """Read a tile ledger, returning an empty compatible ledger when unavailable."""
    ledger_path, _, collection, tile = details
    ledger = {
        "version": 1,
        "collection": collection,
        "tile": tile,
        "files": {},
    }
    if not ledger_path.is_file():
        return ledger

    try:
        loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("checksum ledger root must be an object")
        files = loaded.get("files")
        if loaded.get("version") != 1 or not isinstance(files, dict):
            raise ValueError("unsupported or malformed checksum ledger")
        ledger["files"] = {
            str(key): digest.lower()
            for key, digest in files.items()
            if isinstance(digest, str)
            and len(digest) == 40
            and all(character in "0123456789abcdefABCDEF" for character in digest)
        }
    except (OSError, UnicodeError, ValueError) as exc:
        logger.warning("Invalid checksum ledger %s: %s", ledger_path, exc)
    return ledger


def _write_checksum_ledger(ledger_path, ledger):
    """Durably and atomically write a collection/tile checksum ledger."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ledger_path.with_suffix(f"{ledger_path.suffix}.{random.randint(100000, 999999)}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as stream:
            json.dump(ledger, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                tmp_path.replace(ledger_path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        tmp_path.unlink(missing_ok=True)


def _sha1_file(path, chunk_size=8 * 1024 * 1024):
    """Calculate a file SHA-1 using one sequential pass."""
    digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scrub_cache(
    cache_root,
    checkpoint_file=None,
    corrupt_log=None,
    partition_depth=7,
    progress_interval=60,
):
    """
    Validate every .tif, delete corrupt files, and ledger valid file SHA-1 digests.

    A shallow, sequential walk yields directories at ``partition_depth`` as
    work units. The default depth of 7 is the month directory in the DEA layout::

        dea-public-data/<baseline|derivative>/<collection>/<path>/<row>/<year>/<month>

    Completed units are checkpointed immediately. A restart therefore skips
    completed months without descending into them and repeats at most the
    interrupted month. Sequential reads avoid seek contention on a mechanical
    cache drive.

    Corrupt TIFFs are appended to the TSV log and flushed to disk before they
    are deleted. A unit with a traversal, validation, or deletion error is not
    checkpointed and will be retried by the next run. Valid TIFF checksums are
    written to collection/tile ledgers below ``<cache_root>/checksum-sha1`` before their
    work unit is checkpointed.

    checkpoint_file   - JSON checkpoint; defaults to
                        <cache_root>/audit_checkpoint.json
    corrupt_log       - TSV file (path TAB error); defaults to
                        <cache_root>/audit_corrupt.log
    partition_depth   - work-unit depth (7 = month in the DEA layout)
    progress_interval - seconds between progress messages; 0 disables them
    """
    checkpoint_version = 3
    cache_root = Path(cache_root).resolve()
    if not cache_root.is_dir():
        raise NotADirectoryError(f"Cache root does not exist or is not a directory: {cache_root}")
    if partition_depth < 0:
        raise ValueError("partition_depth must be non-negative")
    if progress_interval < 0:
        raise ValueError("progress_interval must be non-negative")

    checkpoint_file = (
        Path(checkpoint_file) if checkpoint_file else cache_root / "audit_checkpoint.json"
    )
    corrupt_log = Path(corrupt_log) if corrupt_log else cache_root / "audit_corrupt.log"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt_log.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = os.path.normcase(str(cache_root))
    excluded_directories = {"checksum-sha1", "stac-search-cache"}
    checksum_ledgers = {}

    def _work_unit_key(path):
        return path.relative_to(cache_root).as_posix()

    def _ledger_entry(path):
        cache_path = path.relative_to(cache_root).as_posix()
        details = _checksum_ledger_details(cache_root, cache_path)
        if details is None:
            return None, None
        ledger_path = details[0]
        ledger = checksum_ledgers.get(ledger_path)
        if ledger is None:
            ledger = _read_checksum_ledger(details)
            checksum_ledgers[ledger_path] = ledger
        return details, ledger

    completed = set()
    if checkpoint_file.exists():
        try:
            checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read audit checkpoint {checkpoint_file}: {exc}") from exc

        compatible = (
            isinstance(checkpoint, dict)
            and checkpoint.get("version") == checkpoint_version
            and checkpoint.get("cache_root") == checkpoint_root
            and checkpoint.get("partition_depth") == partition_depth
            and isinstance(checkpoint.get("completed_units"), list)
        )
        if compatible:
            completed = set(checkpoint["completed_units"])
            logger.info(
                "audit_cache: resuming with %d work units already verified",
                len(completed),
            )
        else:
            logger.warning(
                "audit_cache: ignoring an incompatible checkpoint at %s",
                checkpoint_file,
            )

    def _save_checkpoint():
        checkpoint = {
            "version": checkpoint_version,
            "cache_root": checkpoint_root,
            "partition_depth": partition_depth,
            "completed_units": sorted(completed),
        }
        tmp = checkpoint_file.with_name(f"{checkpoint_file.name}.tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(checkpoint, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                tmp.replace(checkpoint_file)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))

    def _shallow_walk(root, target_depth):
        """Yield directories at target_depth, or shallower leaf directories."""
        stack = [(root, 0)]
        while stack:
            path, depth = stack.pop()
            if depth == target_depth:
                yield path
                continue
            try:
                with os.scandir(path) as entries:
                    subdirs = sorted(
                        (
                            Path(entry.path)
                            for entry in entries
                            if entry.is_dir(follow_symlinks=False)
                            and entry.name not in excluded_directories
                        ),
                        key=lambda item: item.name.casefold(),
                        reverse=True,
                    )
            except OSError as exc:
                logger.error("audit_cache: cannot enumerate %s: %s", path, exc)
                yield path
                continue
            if subdirs:
                stack.extend((subdir, depth + 1) for subdir in subdirs)
            else:
                yield path

    total_corrupt = 0
    total_deleted = 0
    total_files = 0
    total_checksums = 0
    total_errors = 0
    completed_this_run = 0
    incomplete_this_run = 0
    skipped = 0
    last_progress = time.monotonic()

    def _log_progress(current_path, force=False):
        nonlocal last_progress
        now = time.monotonic()
        if not force and (not progress_interval or now - last_progress < progress_interval):
            return
        logger.info(
            "audit_cache: %d files checked, %d checksums recorded, %d units completed, "
            "%d skipped, %d corrupt deleted, %d errors; current=%s",
            total_files,
            total_checksums,
            completed_this_run,
            skipped,
            total_deleted,
            total_errors,
            current_path,
        )
        last_progress = now

    def _validate_work_unit(work_unit, log_stream):
        nonlocal total_checksums, total_corrupt, total_deleted, total_errors, total_files
        unit_ok = True
        dirty_ledgers = set()

        def _walk_error(exc):
            nonlocal unit_ok, total_errors
            unit_ok = False
            total_errors += 1
            logger.error("audit_cache: cannot traverse %s: %s", work_unit, exc)

        for root, dirs, files in os.walk(work_unit, onerror=_walk_error):
            dirs[:] = sorted(
                (directory for directory in dirs if directory not in excluded_directories),
                key=str.casefold,
            )
            for name in sorted(files, key=str.casefold):
                if Path(name).suffix.casefold() != ".tif":
                    continue

                path = Path(root, name)
                total_files += 1
                try:
                    ok, error = validate_raster(path)
                except Exception as exc:
                    unit_ok = False
                    total_errors += 1
                    logger.exception(
                        "audit_cache: unexpected validation error for %s: %s",
                        path,
                        exc,
                    )
                    _log_progress(path)
                    continue

                if not ok:
                    total_corrupt += 1
                    error = str(error or "unknown raster validation error").replace("\t", " ")
                    error = " ".join(error.splitlines())
                    log_stream.write(f"{path}\t{error}\n")
                    log_stream.flush()
                    os.fsync(log_stream.fileno())
                    try:
                        path.unlink()
                    except OSError as exc:
                        unit_ok = False
                        total_errors += 1
                        logger.error(
                            "audit_cache: cannot delete corrupt TIFF %s: %s",
                            path,
                            exc,
                        )
                    else:
                        total_deleted += 1
                        logger.warning("audit_cache: deleted corrupt TIFF %s: %s", path, error)
                    details, ledger = _ledger_entry(path)
                    if details and ledger["files"].pop(details[1], None) is not None:
                        dirty_ledgers.add(details[0])
                else:
                    details, ledger = _ledger_entry(path)
                    if details is None:
                        logger.debug(
                            "audit_cache: no collection/tile checksum ledger mapping for %s",
                            path,
                        )
                    else:
                        try:
                            digest = _sha1_file(path)
                        except OSError as exc:
                            unit_ok = False
                            total_errors += 1
                            logger.error("audit_cache: cannot checksum %s: %s", path, exc)
                        else:
                            if ledger["files"].get(details[1]) != digest:
                                ledger["files"][details[1]] = digest
                                dirty_ledgers.add(details[0])
                            total_checksums += 1

                _log_progress(path)

        for ledger_path in sorted(dirty_ledgers):
            try:
                _write_checksum_ledger(ledger_path, checksum_ledgers[ledger_path])
            except OSError as exc:
                unit_ok = False
                total_errors += 1
                logger.error("audit_cache: cannot write checksum ledger %s: %s", ledger_path, exc)

        return unit_ok

    # Create the checkpoint and log immediately so the running audit is visible.
    _save_checkpoint()
    with corrupt_log.open("a", encoding="utf-8", buffering=1) as log_stream:
        for work_unit in _shallow_walk(cache_root, partition_depth):
            unit_key = _work_unit_key(work_unit)
            if unit_key in completed:
                skipped += 1
                continue

            if _validate_work_unit(work_unit, log_stream):
                completed.add(unit_key)
                completed_this_run += 1
                _save_checkpoint()
                logger.info("audit_cache: checkpointed completed unit %s", work_unit)
            else:
                incomplete_this_run += 1
                logger.error(
                    "audit_cache: not checkpointing incomplete unit %s; it will be retried",
                    work_unit,
                )

    _log_progress(cache_root, force=True)
    logger.info(
        "audit_cache: pass finished - %d files checked, %d checksums recorded, "
        "%d units completed, %d units incomplete, %d corrupt found, %d deleted; log=%s",
        total_files,
        total_checksums,
        completed_this_run,
        incomplete_this_run,
        total_corrupt,
        total_deleted,
        corrupt_log,
    )
    return total_corrupt


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
                    logger.warning("Insufficient disk space, cache disabled")

            self._url_map = OrderedDict()
            self._inflight = {}
            self._asset_metadata = OrderedDict()
            self._lock = threading.Lock()
            self._manifest_cache = {}
            self._manifest_inflight = {}
            self._manifest_lock = threading.Lock()
            self._checksum_ledgers = {}
            self._dirty_checksum_ledgers = set()
            self._checksum_ledger_lock = threading.RLock()

            self._limiter = RateLimiter(max_rate_mb)

            if patch_url_delay < 0 or patch_url_delay >= 5:
                logger.warning(
                    f"Invalid patch_url_delay={patch_url_delay}, must be >=0 and <5, defaulting to 0.1"
                )
                patch_url_delay = 0.1
            self.patch_url_delay = float(patch_url_delay)

            if url_list_capacity <= 0:
                logger.warning(
                    f"Invalid url_list_capacity={url_list_capacity}, defaulting to 10000"
                )
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
        if "cache_root" not in config:
            cache_root = os.getenv("STAC_CACHE_ROOT")
            if not cache_root:
                raise ValueError(
                    "STAC_CACHE_ROOT environment variable is required. "
                    "Set it in your .env file or pass cache_root= parameter."
                )
            config["cache_root"] = cache_root

        # Build cascading fallback chain
        if "fallback_cache" not in config:
            fallback_chain = cls._build_fallback_chain_from_env()
            if fallback_chain:
                config["fallback_cache"] = fallback_chain

        # Copy-up setting
        if "copy_up" not in config:
            copy_up_env = os.getenv("STAC_CACHE_COPY_UP", "").lower()
            config["copy_up"] = copy_up_env in ("1", "true", "yes")

        # Max workers
        if "max_workers" not in config:
            max_workers_env = os.getenv("STAC_CACHE_MAX_WORKERS")
            if max_workers_env and max_workers_env.isdigit():
                config["max_workers"] = int(max_workers_env)

        # Patch URL delay
        if "patch_url_delay" not in config:
            delay_env = os.getenv("PATCH_URL_DELAY")
            if delay_env:
                try:
                    config["patch_url_delay"] = float(delay_env)
                except ValueError:
                    logger.warning(f"Invalid PATCH_URL_DELAY: {delay_env}")

        # Rate limit
        if "max_rate_mb" not in config:
            rate_env = os.getenv("STAC_CACHE_MAX_RATE_MB")
            if rate_env:
                try:
                    config["max_rate_mb"] = float(rate_env)
                except ValueError:
                    logger.warning(f"Invalid STAC_CACHE_MAX_RATE_MB: {rate_env}")

        # Read-only mode
        if "readonly" not in config:
            readonly_env = os.getenv("STAC_CACHE_READONLY", "").lower()
            config["readonly"] = readonly_env in ("1", "true", "yes")

        # Cache catalog search results
        if "cache_searches" not in config:
            cache_search_env = os.getenv("STAC_CACHE_SEARCHES", "").lower()
            config["cache_searches"] = cache_search_env in ("1", "true", "yes")

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
        fallbacks_env = os.getenv("STAC_CACHE_FALLBACKS")
        if not fallbacks_env:
            return None

        # Split by comma and clean whitespace
        fallback_paths = [p.strip() for p in fallbacks_env.split(",") if p.strip()]

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
                    fallback_cache=chain,  # Link to previous in chain
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
            tmp_path = dest_path.with_suffix(".tmp")
            shutil.copy2(source_path, tmp_path)
            os.replace(tmp_path, dest_path)

            # Update memory map
            with self._lock:
                while len(self._url_map) >= self._url_list_capacity:
                    self._url_map.popitem(last=False)
                self._url_map[url] = str(dest_path)

            file_size = Path(dest_path).stat().st_size / (1 << 20)
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
        return str(path).startswith(("\\\\", "//"))

    @staticmethod
    def _validate_path_with_timeout(path, timeout=5):
        result = {"exists": False}

        def check():
            try:
                result["exists"] = Path(path).is_file()
            except Exception:
                result["exists"] = False

        thread = threading.Thread(target=check, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        return result["exists"]

    @staticmethod
    def _url_to_path(url):
        """
        Normalise any DEA URL form to a local path structure including the bucket name.
        e.g. s3://dea-public-data/... -> dea-public-data/...
        """
        if "data.dea.ga.gov.au" in url:
            return url.replace("https://data.dea.ga.gov.au/", "dea-public-data/")

        if url.startswith("s3://"):
            return url.replace("s3://", "")

        return url

    @staticmethod
    def _https_url(url, cache_path):
        """Convert a DEA S3 asset URL to the public HTTPS endpoint."""
        if cache_path.startswith("dea-public-data/"):
            relative_path = cache_path.removeprefix("dea-public-data/")
            return f"https://data.dea.ga.gov.au/{relative_path}"
        if url.startswith("http"):
            return url
        return f"https://data.dea.ga.gov.au/{cache_path}"

    @staticmethod
    def _parse_sha1_manifest(text):
        """Parse a DEA SHA-1 manifest into a filename-to-digest mapping."""
        checksums = {}
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                digest, filename = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"invalid SHA-1 manifest line {line_number}") from exc
            digest = digest.lower()
            if len(digest) != 40 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"invalid SHA-1 digest on manifest line {line_number}")
            filename = filename.strip().lstrip("*")
            if not filename:
                raise ValueError(f"missing filename on manifest line {line_number}")
            checksums[Path(filename).name] = digest

        if not checksums:
            raise ValueError("SHA-1 manifest contains no file entries")
        return checksums

    def _load_checksum_manifest(self, manifest_url):
        """Fetch and parse a DEA checksum manifest without persisting it in the asset tree."""
        cache_path = self._url_to_path(manifest_url)
        request_url = self._https_url(manifest_url, cache_path)
        headers = {
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
        }
        content = bytearray()
        with requests.get(
            request_url,
            headers=headers,
            stream=True,
            timeout=self._timeout,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                self._limiter.consume(len(chunk))
                content.extend(chunk)

        return self._parse_sha1_manifest(content.decode("utf-8"))

    def _refresh_checksum_manifest(self, manifest_url):
        """Fetch the current DEA manifest, bypassing disk and in-memory copies."""
        try:
            manifest = self._load_checksum_manifest(manifest_url, refresh=True)
        except Exception as exc:
            logger.warning("Unable to refresh SHA-1 manifest %s: %s", manifest_url, exc)
            return None

        with self._manifest_lock:
            self._manifest_cache[manifest_url] = manifest
        return manifest

    def _get_checksum_manifest(self, manifest_url):
        """Return a parsed manifest, allowing only one fetch per URL at a time."""
        with self._manifest_lock:
            if manifest_url in self._manifest_cache:
                return self._manifest_cache[manifest_url]
            event = self._manifest_inflight.get(manifest_url)
            owner = event is None
            if owner:
                event = threading.Event()
                self._manifest_inflight[manifest_url] = event

        if not owner:
            event.wait()
            with self._manifest_lock:
                return self._manifest_cache.get(manifest_url)

        manifest = None
        try:
            manifest = self._load_checksum_manifest(manifest_url)
        except Exception as exc:
            logger.warning("Unable to load SHA-1 manifest %s: %s", manifest_url, exc)
        finally:
            with self._manifest_lock:
                self._manifest_cache[manifest_url] = manifest
                self._manifest_inflight.pop(manifest_url).set()

        return manifest

    def _checksum_ledger_details(self, url):
        """Map a DEA Landsat asset URL to its collection/tile ledger and entry key."""
        return _checksum_ledger_details(self.cache_root, self._url_to_path(url))

    def _load_checksum_ledger(self, details):
        """Load one tile ledger once, returning an empty ledger for missing/invalid files."""
        ledger_path = details[0]
        with self._checksum_ledger_lock:
            ledger = self._checksum_ledgers.get(ledger_path)
            if ledger is not None:
                return ledger

            ledger = _read_checksum_ledger(details)
            self._checksum_ledgers[ledger_path] = ledger
            return ledger

    def _read_asset_checksum(self, url):
        """Read a cached digest without accessing the deeply nested asset directory."""
        details = self._checksum_ledger_details(url)
        if details is None:
            return None
        ledger = self._load_checksum_ledger(details)
        with self._checksum_ledger_lock:
            return ledger["files"].get(details[1])

    def _record_asset_checksum(self, url, digest):
        """Record a validated digest in the in-memory collection/tile ledger."""
        details = self._checksum_ledger_details(url)
        if details is None:
            logger.debug("No collection/tile checksum ledger mapping for %s", url)
            return
        ledger = self._load_checksum_ledger(details)
        with self._checksum_ledger_lock:
            ledger["files"][details[1]] = digest.lower()
            self._dirty_checksum_ledgers.add(details[0])

    def _remove_asset_checksum(self, url):
        """Remove a stale digest from its in-memory ledger."""
        details = self._checksum_ledger_details(url)
        if details is None:
            return
        ledger = self._load_checksum_ledger(details)
        with self._checksum_ledger_lock:
            if ledger["files"].pop(details[1], None) is not None:
                self._dirty_checksum_ledgers.add(details[0])

    def _flush_checksum_ledgers(self):
        """Atomically flush dirty tile ledgers, normally once the download queue is idle."""
        if self.readonly:
            return
        with self._checksum_ledger_lock:
            for ledger_path in list(self._dirty_checksum_ledgers):
                ledger = self._checksum_ledgers[ledger_path]
                try:
                    _write_checksum_ledger(ledger_path, ledger)
                    self._dirty_checksum_ledgers.discard(ledger_path)
                except OSError as exc:
                    logger.warning("Unable to write checksum ledger %s: %s", ledger_path, exc)

    def _reconcile_asset_checksums(self, assets):
        """Remove cached TIFFs only when fresh and recorded SHA-1 digests disagree."""
        manifest_urls = sorted({asset[2] for asset in assets if asset[2]})
        manifests = {}

        if manifest_urls:
            worker_count = min(max(1, self.workers), 8, len(manifest_urls))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(self._refresh_checksum_manifest, url): url
                    for url in manifest_urls
                }
                for future in as_completed(futures):
                    manifest_url = futures[future]
                    try:
                        manifests[manifest_url] = future.result()
                    except Exception as exc:
                        logger.warning(
                            "Unable to reconcile SHA-1 manifest %s: %s", manifest_url, exc
                        )
                        manifests[manifest_url] = None

        reconciled = []
        for priority, url, checksum_url, expected_size in assets:
            local_path = self.cache_root / Path(self._url_to_path(url))
            manifest = manifests.get(checksum_url)
            expected_sha1 = manifest.get(local_path.name) if manifest else None
            cached_sha1 = self._read_asset_checksum(url)

            if expected_sha1 and cached_sha1 and expected_sha1 != cached_sha1:
                try:
                    local_path.unlink(missing_ok=True)
                    self._remove_asset_checksum(url)
                    with self._lock:
                        self._url_map.pop(url, None)
                    logger.warning(
                        "Deleted stale cached TIFF %s: cached SHA-1 %s differs from DEA %s",
                        local_path,
                        cached_sha1,
                        expected_sha1,
                    )
                except OSError as exc:
                    logger.error("Unable to delete stale cached TIFF %s: %s", local_path, exc)

            with self._lock:
                metadata = self._asset_metadata.get(url)
                if metadata is not None:
                    metadata["expected_sha1"] = expected_sha1
            reconciled.append((priority, url, checksum_url, expected_size, expected_sha1))

        return reconciled

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

    def _download_asset(
        self,
        url,
        priority=None,
        max_retries=3,
        checksum_url=None,
        expected_size=None,
        expected_sha1=None,
    ):
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

            if expected_size is not None:
                try:
                    expected_size = int(expected_size)
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring invalid expected size for %s: %r", s3_path, expected_size
                    )
                    expected_size = None
            if expected_sha1 is not None:
                expected_sha1 = str(expected_sha1).lower()
            elif checksum_url:
                manifest = self._get_checksum_manifest(checksum_url)
                if manifest:
                    expected_sha1 = manifest.get(local_dest.name)
                    if not expected_sha1:
                        logger.warning(
                            "SHA-1 manifest has no entry for %s; using lightweight open check",
                            s3_path,
                        )

            tmp_dest = local_dest.with_suffix(f".{random.randint(100000, 999999)}.tmp")
            https_url = self._https_url(url, s3_path)

            for attempt in range(max_retries):
                try:
                    p_str = f" [p={priority}]" if priority is not None else ""
                    log = logger.warning if attempt > 1 else logger.debug
                    log(
                        f"Downloading{p_str}: {s3_path} (Attempt {attempt + 1}) "
                        f"[thread={threading.current_thread().name}]"
                    )
                    t0 = time.time()

                    request_url = https_url

                    if attempt > 0:
                        request_url = f"{https_url}?cachebreaker={attempt}"

                    headers = {
                        "Cache-Control": "no-cache, no-store, max-age=0",
                        "Pragma": "no-cache",
                    }

                    # 10s connect timeout, 300s read timeout
                    # Read timeout is generous: 24 threads sharing ~100 Mbps means
                    # each thread gets ~4 Mbps — a 50 MB file can take 100s+ between chunks.
                    chunk_sz = 1024 * 1024 if self._limiter.rate_per_sec > 0 else 8 * 1024 * 1024
                    sha1 = hashlib.sha1(usedforsecurity=False) if expected_sha1 else None
                    downloaded_bytes = 0
                    with requests.get(
                        request_url, headers=headers, stream=True, timeout=self._timeout
                    ) as resp:
                        resp.raise_for_status()
                        with open(tmp_dest, "wb") as local_f:
                            for chunk in resp.iter_content(chunk_size=chunk_sz):
                                if not chunk:
                                    continue

                                self._limiter.consume(len(chunk))
                                local_f.write(chunk)
                                downloaded_bytes += len(chunk)
                                if sha1:
                                    sha1.update(chunk)

                    if expected_size is not None and downloaded_bytes != expected_size:
                        raise OSError(
                            f"Downloaded size mismatch for {s3_path}: "
                            f"expected {expected_size}, got {downloaded_bytes} bytes"
                        )

                    if expected_sha1:
                        actual_sha1 = sha1.hexdigest()
                        if actual_sha1 != expected_sha1:
                            raise OSError(
                                f"SHA-1 mismatch for {s3_path}: "
                                f"expected {expected_sha1}, got {actual_sha1}"
                            )
                        logger.debug("SHA-1 verified: %s", s3_path)
                    else:
                        valid, error = validate_raster_open(tmp_dest)
                        if not valid:
                            raise OSError(f"Raster open validation failed for {s3_path}: {error}")
                        logger.info(
                            "No SHA-1 checksum available for %s; "
                            "passed lightweight raster open check",
                            s3_path,
                        )

                    # Atomic rename — no lock needed: tmp name is unique per thread,
                    # and replace() is atomic on NTFS.
                    tmp_dest.replace(local_dest)
                    if expected_sha1:
                        self._record_asset_checksum(url, expected_sha1)
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
                        time.sleep(2**attempt)

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
            queue_idle = not self._inflight
        if queue_idle:
            self._flush_checksum_ledgers()
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

            checksum_asset = item.assets.get("checksum:sha1")
            checksum_url = getattr(checksum_asset, "href", None)
            asset_keys = bands if bands is not None else list(item.assets.keys())
            for band in asset_keys:
                if band == "checksum:sha1":
                    continue
                if band in item.assets:
                    url = item.assets[band].href
                    if url not in seen:
                        seen.add(url)
                        size = item.assets[band].extra_fields.get("file:size")
                        try:
                            size = int(size) if size is not None else None
                        except (TypeError, ValueError):
                            logger.warning("Ignoring invalid file:size for %s: %r", url, size)
                            size = None
                        assets.append((priority, url, checksum_url, size))
                        with self._lock:
                            while (
                                len(self._asset_metadata) >= self._url_list_capacity
                                and url not in self._asset_metadata
                            ):
                                self._asset_metadata.popitem(last=False)
                            self._asset_metadata[url] = {
                                "checksum_url": checksum_url,
                                "expected_size": size,
                            }
                            self._asset_metadata.move_to_end(url)
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

        total_expected = (
            f" (~{total_expected_bytes / (1024 * 1024):.1f} MB expected)"
            if total_expected_bytes > 0
            else ""
        )

        logger.info(
            f"Starting {len(assets_to_download)} downloads with {self.workers} workers{total_expected}"
        )
        for i in assets_to_download:
            logger.debug(f"Priority: {i[0]}, URL: {i[1]}")

        for priority, url, _checksum_url, expected_size, expected_sha1 in assets_to_download:
            with self._lock:
                # Skip already cached
                if url in self._url_map:
                    continue
                # Skip if already queued or running
                if url in self._inflight:
                    continue
                # Submit download with priority metadata
                fut = self._executor.submit(
                    self._download_asset,
                    url,
                    priority=priority,
                    expected_size=expected_size,
                    expected_sha1=expected_sha1,
                )
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
            logger.warning(
                f"Invalid 'items' type for caching. Expected iterable, got {type(items)}."
            )
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
        assets = self._reconcile_asset_checksums(assets)
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
                isinstance(second, (list, tuple))
                and (len(second) == 0 or isinstance(second[0], str))
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

        all_assets = self._reconcile_asset_checksums(all_assets)
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
            logger.debug(
                f"Found in fallback: {url} -> {fallback_path}, copy_up={self.copy_up}, readonly={self.readonly}"
            )
            if self.copy_up and not self.readonly:
                # Copy to this cache for faster access next time
                new_path = self._copy_from_fallback(fallback_path, url)
                return new_path
            else:
                # Use directly from fallback
                logger.debug("Using fallback directly (copy_up disabled or readonly)")
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
                metadata = self._asset_metadata.get(url, {})
                fut = self._executor.submit(
                    self._download_asset,
                    url,
                    expected_size=metadata.get("expected_size"),
                    expected_sha1=metadata.get("expected_sha1"),
                )
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

    def cleanup_old_files(cache_root, max_age_days=90):
        """
        Remove files older than max_age_days.

        With noatime: This removes files by download age, not access age.
        Good enough for: "Delete anything downloaded >90 days ago"
        """
        cutoff = time.time() - (max_age_days * 86400)

        deleted = 0
        for filepath in cache_root.rglob("*.tif"):
            if filepath.stat().st_mtime < cutoff:
                try:
                    filepath.unlink()
                    deleted += 1
                except OSError:
                    pass

        return deleted

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
                if not self.readonly and items and hasattr(items, "__len__") and len(items) > 0:
                    try:
                        tmp_file = cache_file.with_suffix(".tmp")
                        with gzip.open(tmp_file, "wt", encoding="utf-8") as f:
                            json.dump(items.to_dict(), f)
                        tmp_file.replace(cache_file)
                        logger.debug(f"Cached {len(items)} items to {cache_file.name}")
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
        """Finish downloads, persist checksum ledgers, and shut down the executor."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=True)
            logger.info("STACCache executor shut down")
        if hasattr(self, "_dirty_checksum_ledgers"):
            self._flush_checksum_ledgers()

    def __enter__(self):
        """Support 'with STACCache() as cache:' syntax."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure clean shutdown on context exit."""
        self.close()
        return False


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    scrub_cache("z:/dea-local-cache")


if __name__ == "__main__":
    main()
