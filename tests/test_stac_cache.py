import hashlib
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to sys.path to allow imports from tools/
current_path = Path(__file__).resolve().parent
project_root = current_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.stac_cache import STACCache, scrub_cache


def test_audit_cache_deletes_corrupt_tiff_and_resumes(tmp_path):
    month = (
        tmp_path / "dea-public-data" / "baseline" / "ga_ls5t_ard_3" / "089" / "077" / "1989" / "03"
    )
    day = month / "24"
    day.mkdir(parents=True)
    good = day / "good.tif"
    corrupt = day / "corrupt.TIF"
    good.write_bytes(b"good")
    corrupt.write_bytes(b"corrupt")

    def validation_result(path):
        if Path(path) == corrupt:
            return False, "broken raster"
        return True, None

    with patch("tools.stac_cache.validate_raster", side_effect=validation_result) as validate:
        assert scrub_cache(tmp_path, progress_interval=0) == 1

    assert validate.call_count == 2
    assert good.exists()
    assert not corrupt.exists()
    assert f"{corrupt}\tbroken raster" in (tmp_path / "audit_corrupt.log").read_text()

    checkpoint = json.loads((tmp_path / "audit_checkpoint.json").read_text())
    assert checkpoint["version"] == 3
    assert checkpoint["partition_depth"] == 7
    assert checkpoint["completed_units"] == [month.relative_to(tmp_path).as_posix()]
    ledger_path = tmp_path / "checksum-sha1/dea-public-data/baseline/ga_ls5t_ard_3" / "089_077.json"
    ledger = json.loads(ledger_path.read_text())
    assert ledger["files"] == {
        "1989/03/24/good.tif": hashlib.sha1(b"good", usedforsecurity=False).hexdigest()
    }

    with patch("tools.stac_cache.validate_raster") as validate:
        assert scrub_cache(tmp_path, progress_interval=0) == 0
    validate.assert_not_called()


def test_audit_cache_retries_unit_after_unexpected_validation_error(tmp_path):
    month = tmp_path / "one" / "two" / "three" / "four" / "five" / "six" / "seven"
    month.mkdir(parents=True)
    raster = month / "image.tif"
    raster.write_bytes(b"data")

    with patch("tools.stac_cache.validate_raster", side_effect=RuntimeError("unexpected")):
        assert scrub_cache(tmp_path, progress_interval=0) == 0

    checkpoint_path = tmp_path / "audit_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["completed_units"] == []

    with patch("tools.stac_cache.validate_raster", return_value=(True, None)) as validate:
        assert scrub_cache(tmp_path, progress_interval=0) == 0
    validate.assert_called_once_with(raster)


def test_audit_cache_does_not_checkpoint_when_checksum_ledger_write_fails(tmp_path):
    month = (
        tmp_path / "dea-public-data" / "baseline" / "ga_ls5t_ard_3" / "089" / "077" / "1989" / "03"
    )
    day = month / "24"
    day.mkdir(parents=True)
    raster = day / "good.tif"
    raster.write_bytes(b"good")

    with (
        patch("tools.stac_cache.validate_raster", return_value=(True, None)),
        patch("tools.stac_cache._write_checksum_ledger", side_effect=OSError("disk error")),
    ):
        assert scrub_cache(tmp_path, progress_interval=0) == 0

    checkpoint = json.loads((tmp_path / "audit_checkpoint.json").read_text())
    assert checkpoint["completed_units"] == []


class TestSTACCache:
    @pytest.fixture
    def cache(self, tmp_path):
        # Initialize cache with 0MB min free space to ensure tests run regardless of disk space
        return STACCache(cache_root=tmp_path, min_free_space_mb=0)

    def test_url_to_path(self):
        """Test URL normalization logic."""
        # DEA HTTPS URL
        url = "https://data.dea.ga.gov.au/baseline/ga_ls8c_ard_3/090/084/2021/06/27/file.tif"
        expected = "dea-public-data/baseline/ga_ls8c_ard_3/090/084/2021/06/27/file.tif"
        assert STACCache._url_to_path(url) == expected

        # S3 URL
        url_s3 = "s3://dea-public-data/baseline/file.tif"
        expected_s3 = "dea-public-data/baseline/file.tif"
        assert STACCache._url_to_path(url_s3) == expected_s3

        # Generic URL
        url_other = "https://example.com/file.tif"
        assert STACCache._url_to_path(url_other) == url_other

    @patch("tools.stac_cache.requests.get")
    def test_download_asset_success(self, mock_get, cache, tmp_path, caplog):
        """Test successful download of an asset."""
        # Mock response context manager
        mock_response = MagicMock()
        mock_response.status_code = 200
        content = b"test_data"
        mock_response.iter_content.return_value = [content]
        mock_get.return_value.__enter__.return_value = mock_response

        url = "https://data.dea.ga.gov.au/baseline/test_success.tif"

        # Execute
        with (
            caplog.at_level(logging.INFO, logger="tools.stac_cache"),
            patch("tools.stac_cache.validate_raster_open", return_value=(True, None)) as validate,
        ):
            original_url, local_path = cache._download_asset(url)

        # Verify return values
        assert original_url == url
        validate.assert_called_once()
        assert Path(local_path).exists()
        assert Path(local_path).read_bytes() == content
        assert "No SHA-1 checksum available" in caplog.text
        assert "test_success.tif" in caplog.text

        # Verify directory structure
        expected_rel_path = "dea-public-data/baseline/test_success.tif"
        # Normalize separators for Windows/Linux compatibility in assertion
        assert str(Path(local_path).relative_to(tmp_path)).replace("\\", "/") == expected_rel_path

    @patch("tools.stac_cache.requests.get")
    def test_download_asset_uses_dea_sha1_manifest(self, mock_get, cache, tmp_path):
        content = b"valid raster bytes"
        digest = hashlib.sha1(content, usedforsecurity=False).hexdigest()
        filename = "ga_ls5t_nbart_3-2-1_089077_1989-03-24_final_band04.tif"
        manifest_name = "tile.sha1"
        manifest_content = f"{digest}\t{filename}\n".encode()

        manifest_response = MagicMock()
        manifest_response.iter_content.return_value = [manifest_content]
        manifest_request = MagicMock()
        manifest_request.__enter__.return_value = manifest_response

        asset_response = MagicMock()
        asset_response.iter_content.return_value = [content]
        asset_request = MagicMock()
        asset_request.__enter__.return_value = asset_response
        mock_get.side_effect = [manifest_request, asset_request]

        tile_url = "s3://dea-public-data/baseline/ga_ls5t_ard_3/089/077/1989/03/24"
        url = f"{tile_url}/{filename}"
        manifest_url = f"{tile_url}/{manifest_name}"
        with patch("tools.stac_cache.validate_raster_open") as validate:
            original_url, local_path = cache._download_asset(
                url,
                checksum_url=manifest_url,
                expected_size=len(content),
            )

        assert original_url == url
        assert Path(local_path).read_bytes() == content
        cache._flush_checksum_ledgers()
        ledger_path = tmp_path / "checksum-sha1/dea-public-data/baseline/ga_ls5t_ard_3" / "089_077.json"
        ledger = json.loads(ledger_path.read_text())
        assert ledger["files"] == {f"1989/03/24/{filename}": digest}
        assert not Path(f"{local_path}checksum-sha1").exists()
        assert not (Path(local_path).parent / manifest_name).exists()
        validate.assert_not_called()
        assert mock_get.call_count == 2

        # The provider manifest remains in memory; only validated digests are persisted.
        assert cache._get_checksum_manifest(manifest_url)[filename] == digest
        assert mock_get.call_count == 2

    @patch("tools.stac_cache.requests.get")
    def test_download_asset_rejects_sha1_mismatch(self, mock_get, cache, tmp_path, caplog):
        expected_content = b"expected"
        downloaded_content = b"corrupt"
        filename = "ga_ls5t_nbart_3-2-1_089077_1989-03-24_final_band04.tif"
        digest = hashlib.sha1(expected_content, usedforsecurity=False).hexdigest()
        manifest_content = f"{digest}\t{filename}\n".encode()

        manifest_response = MagicMock()
        manifest_response.iter_content.return_value = [manifest_content]
        manifest_request = MagicMock()
        manifest_request.__enter__.return_value = manifest_response

        asset_response = MagicMock()
        asset_response.iter_content.return_value = [downloaded_content]
        asset_request = MagicMock()
        asset_request.__enter__.return_value = asset_response
        mock_get.side_effect = [manifest_request, asset_request]

        tile_url = "s3://dea-public-data/baseline/ga_ls5t_ard_3/089/077/1989/03/24"
        url = f"{tile_url}/{filename}"
        manifest_url = f"{tile_url}/tile.sha1"
        with (
            caplog.at_level(logging.WARNING, logger="tools.stac_cache"),
            patch("tools.stac_cache.validate_raster_open") as validate,
        ):
            original_url, local_path = cache._download_asset(
                url,
                max_retries=1,
                checksum_url=manifest_url,
            )

        assert original_url == url
        assert local_path == url
        assert "SHA-1 mismatch" in caplog.text
        assert not (
            tmp_path / "dea-public-data/baseline/ga_ls5t_ard_3/089/077/1989/03/24" / filename
        ).exists()
        validate.assert_not_called()

    @patch("tools.stac_cache.requests.get")
    def test_download_asset_cache_hit(self, mock_get, cache, tmp_path):
        """Test that existing files are not re-downloaded."""
        url = "https://data.dea.ga.gov.au/baseline/test_hit.tif"
        rel_path = "dea-public-data/baseline/test_hit.tif"

        # Create dummy existing file
        file_path = tmp_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"existing_data")

        # Execute
        original_url, local_path = cache._download_asset(url)

        # Verify
        mock_get.assert_not_called()
        assert local_path == str(file_path.resolve())

    @patch("tools.stac_cache.requests.get")
    def test_download_asset_retry_failure(self, mock_get, cache):
        """Test retry logic and failure handling."""
        mock_get.side_effect = Exception("Connection refused")

        url = "https://data.dea.ga.gov.au/baseline/fail.tif"

        # Execute with 2 retries
        original_url, local_path = cache._download_asset(url, max_retries=2)

        # Verify
        assert local_path == url  # Should return original URL on failure
        assert mock_get.call_count == 2

    def test_submit_cache_items(self, cache):
        """Test bulk caching submission logic."""
        # Mock STAC items
        item1 = MagicMock()
        item1.assets = {
            "band1": MagicMock(
                href="https://data.dea.ga.gov.au/b1.tif", extra_fields={"file:size": 100}
            ),
            "checksum:sha1": MagicMock(href="s3://dea-public-data/item1.sha1"),
        }
        item2 = MagicMock()
        item2.assets = {
            "band1": MagicMock(
                href="https://data.dea.ga.gov.au/b2.tif", extra_fields={"file:size": 100}
            ),
        }
        items = [item1, item2]

        # Mock the executor to verify submission
        cache._executor = MagicMock()

        digest = "1" * 40
        with patch.object(
            cache,
            "_refresh_checksum_manifest",
            return_value={"b1.tif": digest},
        ):
            cache.submit_cache_items(items, bands=["band1"])

        # Verify
        # _submit_assets calls executor.submit for each asset
        assert cache._executor.submit.call_count == 2
        cache._executor.submit.assert_any_call(
            cache._download_asset,
            "https://data.dea.ga.gov.au/b1.tif",
            priority=0,
            expected_size=100,
            expected_sha1=digest,
        )

    @pytest.mark.parametrize(
        ("cached_digest", "should_delete"),
        [
            ("a" * 40, False),
            ("b" * 40, True),
            (None, False),
        ],
        ids=["matching", "different", "legacy-without-ledger-entry"],
    )
    def test_reconcile_asset_checksum_policy(self, cache, tmp_path, cached_digest, should_delete):
        filename = "ga_ls5t_nbart_3-2-1_089077_1989-03-24_final_band04.tif"
        tile_url = "s3://dea-public-data/baseline/ga_ls5t_ard_3/089/077/1989/03/24"
        url = f"{tile_url}/{filename}"
        manifest_url = f"{tile_url}/tile.sha1"
        local_path = (
            tmp_path / "dea-public-data/baseline/ga_ls5t_ard_3/089/077/1989/03/24" / filename
        )
        local_path.parent.mkdir(parents=True)
        local_path.write_bytes(b"cached")
        if cached_digest:
            cache._record_asset_checksum(url, cached_digest)

        assets = [(0, url, manifest_url, None)]
        with patch.object(
            cache,
            "_refresh_checksum_manifest",
            return_value={local_path.name: "a" * 40},
        ):
            reconciled = cache._reconcile_asset_checksums(assets)

        assert local_path.exists() is not should_delete
        assert cache._read_asset_checksum(url) == (cached_digest if not should_delete else None)
        assert reconciled == [(0, url, manifest_url, None, "a" * 40)]

    def test_submit_deletes_stale_tiff_before_queueing_replacement(self, cache, tmp_path):
        filename = "ga_ls5t_nbart_3-2-1_089077_1989-03-24_final_band04.tif"
        tile_url = "s3://dea-public-data/baseline/ga_ls5t_ard_3/089/077/1989/03/24"
        url = f"{tile_url}/{filename}"
        manifest_url = f"{tile_url}/tile.sha1"
        local_path = (
            tmp_path / "dea-public-data/baseline/ga_ls5t_ard_3/089/077/1989/03/24" / filename
        )
        local_path.parent.mkdir(parents=True)
        local_path.write_bytes(b"old")
        cache._record_asset_checksum(url, "1" * 40)
        cache._url_map[url] = str(local_path)

        item = MagicMock()
        item.assets = {
            "band": MagicMock(href=url, extra_fields={}),
            "checksum:sha1": MagicMock(href=manifest_url),
        }
        cache._executor = MagicMock()

        with patch.object(
            cache,
            "_refresh_checksum_manifest",
            return_value={local_path.name: "2" * 40},
        ):
            cache.submit_cache_items([item], bands=["band"])

        assert not local_path.exists()
        assert cache._read_asset_checksum(url) is None
        assert url not in cache._url_map
        cache._executor.submit.assert_called_once_with(
            cache._download_asset,
            url,
            priority=0,
            expected_size=None,
            expected_sha1="2" * 40,
        )

    def test_checksum_ledger_consolidates_years_for_collection_tile(self, cache, tmp_path):
        base_url = "s3://dea-public-data/baseline/ga_ls5t_ard_3/089/077"
        first_url = f"{base_url}/1989/03/24/first.tif"
        second_url = f"{base_url}/1990/04/25/second.tif"

        cache._record_asset_checksum(first_url, "1" * 40)
        cache._record_asset_checksum(second_url, "2" * 40)
        cache._flush_checksum_ledgers()

        ledger_path = tmp_path / "checksum-sha1/dea-public-data/baseline/ga_ls5t_ard_3" / "089_077.json"
        ledger = json.loads(ledger_path.read_text())
        assert ledger["files"] == {
            "1989/03/24/first.tif": "1" * 40,
            "1990/04/25/second.tif": "2" * 40,
        }
        assert list((tmp_path / "checksum-sha1").rglob("*.json")) == [ledger_path]

    def test_patch_url_memory_hit(self, cache, tmp_path):
        """Test patch_url returns local path if in memory map."""
        url = "http://example.com/1.tif"
        local_path = tmp_path / "1.tif"
        local_path.touch()

        cache._url_map[url] = str(local_path)
        assert cache.patch_url(url) == str(local_path)

    def test_patch_url_disk_hit(self, cache, tmp_path):
        """Test patch_url returns local path if file exists on disk."""
        url = "https://data.dea.ga.gov.au/baseline/test.tif"
        # _url_to_path converts this to dea-public-data/baseline/test.tif
        rel_path = "dea-public-data/baseline/test.tif"
        local_file = tmp_path / rel_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_bytes(b"data")

        # Ensure map is empty
        cache._url_map.clear()

        result = cache.patch_url(url)
        assert result == str(local_file.resolve())
        # Should populate map
        assert cache._url_map[url] == str(local_file.resolve())

    def test_patch_url_miss_triggers_download(self, cache):
        """Test patch_url triggers download on miss."""
        url = "http://example.com/miss.tif"
        cache._executor = MagicMock()

        # Call with short timeout
        result = cache.patch_url(url, timeout=0.01)

        assert result == url
        cache._executor.submit.assert_called_once()
        assert url in cache._inflight

    def test_patch_url_inflight(self, cache):
        """Test patch_url respects inflight status."""
        url = "http://example.com/inflight.tif"
        cache._inflight[url] = (MagicMock(), None)
        cache._executor = MagicMock()

        result = cache.patch_url(url, timeout=0.01)

        assert result == url
        cache._executor.submit.assert_not_called()

    @patch("tools.stac_cache.time.sleep")
    def test_patch_url_wait_success(self, mock_sleep, cache, tmp_path):
        """Test patch_url returns path if download completes during timeout wait."""
        url = "http://example.com/wait.tif"
        dest = tmp_path / "wait.tif"
        dest.touch()

        # Define side effect for sleep to simulate download completing
        def sleep_side_effect(seconds):
            cache._url_map[url] = str(dest)

        mock_sleep.side_effect = sleep_side_effect

        # Mock executor
        cache._executor = MagicMock()

        result = cache.patch_url(url, timeout=1)

        assert result == str(dest)
        cache._executor.submit.assert_called_once()
