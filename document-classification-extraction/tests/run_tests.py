#!/usr/bin/env python
"""
Test runner script for document classification and extraction system.
"""

import unittest
import sys
import os

def run_tests():
    """
    Discover and run all tests.
    
    Returns:
        int: 0 if all tests pass, non-zero otherwise
    """
    # Ensure the package root is in the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Load test cases from tests directory
    loader = unittest.TestLoader()
    test_suite = loader.discover('tests', pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return 0 if tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    print("Running Document Classification and Extraction System Tests")
    print("-" * 70)
    sys.exit(run_tests())