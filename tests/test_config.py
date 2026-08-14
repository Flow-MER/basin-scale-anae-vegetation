#!/usr/bin/env python3
"""
Test script for the new YAML-based configuration system.
Verifies that all configurations can be loaded and validated correctly.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config


def test_config_loading():
    """Test loading all configuration types."""
    config_names = ["ndvi_landsat", "soil_moisture", "wit_metrics", "vegetation"]

    print("Testing configuration loading...")

    for config_name in config_names:
        try:
            config = load_config(config_name)
            print(f"✓ {config_name}: Loaded successfully")

            # Test some basic attributes
            assert hasattr(config, "data_path"), f"{config_name} missing data_path"
            assert hasattr(config, "log_path"), f"{config_name} missing log_path"
            assert hasattr(config, "shapefile_path"), f"{config_name} missing shapefile_path"

            # Test path resolution
            assert config.data_path.is_absolute(), f"{config_name} data_path not absolute"
            assert config.log_path.is_absolute(), f"{config_name} log_path not absolute"

            print(f"  - data_path: {config.data_path}")
            print(f"  - log_path: {config.log_path}")

        except Exception as e:
            print(f"✗ {config_name}: Failed to load - {e}")
            pytest.fail(f"{config_name}: Failed to load - {e}")


def test_config_manager():
    """Test the config manager directly."""
    print("\nTesting config manager...")

    try:
        from config import CONFIG_MODELS

        print(f"✓ Config models available: {list(CONFIG_MODELS.keys())}")
    except Exception as e:
        print(f"✗ Config manager: Failed - {e}")
        pytest.fail(f"Config manager: Failed - {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Configuration System Test")
    print("=" * 60)

    tests = [test_config_loading, test_config_manager]

    results = []
    for test in tests:
        try:
            test()
            results.append(True)
        except AssertionError as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            results.append(False)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} tests passed!")
        return 0
    else:
        print(f"✗ {passed}/{total} tests passed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
