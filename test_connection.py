"""Smoke test: can we reach Snowflake with the credentials in .env?

Run from the project root:
    python test_connection.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from db.snowflake_client import execute_query, test_connection  # noqa: E402


def main() -> int:
    if os.environ.get("SNOWFLAKE_PASSWORD", "").startswith("PLACEHOLDER"):
        print(
            "ERROR: SNOWFLAKE_PASSWORD is still a placeholder in .env. "
            "Fill it in before running this script."
        )
        return 1

    print("Testing connection...")
    ok = test_connection()
    print("Connected:", ok)

    if not ok:
        return 2

    rows = execute_query(
        "SELECT TABLE_NAME FROM "
        "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET"
        ".INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = 'PUBLIC' LIMIT 10"
    )
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
