#!/usr/bin/env python3
"""
Test script to verify local dataset loading functionality.
This script tests the modified task classes without requiring datasets library.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test that modified task files can be imported (without datasets dependency)."""
    try:
        # Mock the datasets module to avoid import errors
        import sys
        from unittest.mock import MagicMock

        # Create mock for datasets
        datasets_mock = MagicMock()
        sys.modules['datasets'] = datasets_mock

        # Test imports
        from tasks.smoltalk import SmolTalk
        from tasks.mmlu import MMLU
        from tasks.humaneval import HumanEval
        from tasks.gsm8k import GSM8K
        from tasks.arc import ARC

        print("✓ All task classes imported successfully")

        # Test that classes have the expected signatures
        assert hasattr(SmolTalk.__init__, '__code__')
        assert 'data_dir' in SmolTalk.__init__.__code__.co_varnames

        assert hasattr(MMLU.__init__, '__code__')
        assert 'data_dir' in MMLU.__init__.__code__.co_varnames

        assert hasattr(HumanEval.__init__, '__code__')
        assert 'data_dir' in HumanEval.__init__.__code__.co_varnames

        assert hasattr(GSM8K.__init__, '__code__')
        assert 'data_dir' in GSM8K.__init__.__code__.co_varnames

        assert hasattr(ARC.__init__, '__code__')
        assert 'data_dir' in ARC.__init__.__code__.co_varnames

        print("✓ All task classes have data_dir parameter")

        return True

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_data_structure():
    """Test that data directory exists."""
    data_dir = Path("./data")

    if not data_dir.exists():
        print("✗ Data directory does not exist")
        return False

    print("✓ Data directory exists")
    return True

def main():
    print("Testing local dataset loading modifications...")
    print()

    success = True

    # Test imports
    if not test_import():
        success = False

    # Test data directory structure
    if not test_data_structure():
        success = False

    print()
    if success:
        print("🎉 All tests passed!")
        print()
        print("Usage:")
        print("1. Run 'python scripts/download_datasets.py' to download all datasets")
        print("2. Use data_dir='./data' parameter when creating task instances:")
        print("   task = SmolTalk(split='train', data_dir='./data')")
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
