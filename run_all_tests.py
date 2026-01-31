import sys
import pytest
from pathlib import Path

def main():
    """
    Discover and run all tests in the 'tests' directory using pytest.
    This script serves as the primary entry point for the project's test suite.
    """
    # Define paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    tests_dir = project_root / "tests"

    print(f"Starting test suite run for FlowMER2.0...")
    print(f"Discovering tests in: {tests_dir}\n")

    # Run pytest programmatically
    # -v: Verbose output
    # str(tests_dir): Target directory
    exit_code = pytest.main(["-v", str(tests_dir)])

    sys.exit(exit_code)

if __name__ == "__main__":
    main()