"""Shared pytest fixtures.

Sets dummy environment variables so modules that read os.environ at import
time don't blow up, and ensures tests never touch real services.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the project root importable without installing as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test_account")
    monkeypatch.setenv("SNOWFLAKE_USER", "test_user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "test_pw")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "TEST_WH")
    monkeypatch.setenv(
        "SNOWFLAKE_DATABASE",
        "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
    )
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "PUBLIC")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    yield
