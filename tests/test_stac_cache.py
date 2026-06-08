import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to sys.path to allow imports from tools/
current_path = Path(__file__).resolve().parent
project_root = current_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.stac_cache import STACCache


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
    def test_download_asset_success(self, mock_get, cache, tmp_path):
        """Test successful download of an asset."""
        # Mock response context manager
        mock_response = MagicMock()
        mock_response.status_code = 200
        content = b"test_data"
        mock_response.iter_content.return_value = [content]
        mock_get.return_value.__enter__.return_value = mock_response

        url = "https://data.dea.ga.gov.au/baseline/test_success.tif"

        # Execute
        original_url, local_path = cache._download_asset(url)

        # Verify return values
        assert original_url == url
        assert Path(local_path).exists()
        assert Path(local_path).read_bytes() == content

        # Verify directory structure
        expected_rel_path = "dea-public-data/baseline/test_success.tif"
        # Normalize separators for Windows/Linux compatibility in assertion
        assert str(Path(local_path).relative_to(tmp_path)).replace("\\", "/") == expected_rel_path

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

        # Execute
        cache.submit_cache_items(items, bands=["band1"])

        # Verify
        # _submit_assets calls executor.submit for each asset
        assert cache._executor.submit.call_count == 2

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
