"""
Pytest configuration and shared fixtures for unit tests.
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test API key if not already set
    if 'GROQ_API_KEY' not in os.environ:
        os.environ['GROQ_API_KEY'] = 'test_key_for_unit_tests'
    
    yield
    
    # Cleanup (if needed)
    pass


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data files."""
    return tmp_path


@pytest.fixture
def sample_restaurant_data():
    """Sample restaurant data for testing."""
    return {
        "id": 1,
        "name": "Test Restaurant",
        "cuisine": "Italian",
        "location": "Downtown Dubai",
        "price_range": "AED 100-200",
        "description": "A test restaurant",
        "amenities": "WiFi, Parking",
        "attributes": "Romantic",
        "opening_hours": "10:00-22:00",
        "rating": 4.5,
        "review_count": 100
    }

