"""Build the compact schema-context string the SQL-generation LLM sees.

The Census database has hundreds of tables and tens of thousands of
columns, which is way too much for a prompt. We produce a <3000-char
summary with:
  - a short legend of ACS table codes (B01 = age/sex, ...)
  - an inventory of available `<YEAR>_CBG_<CODE>` tables
  - the column list for the metadata tables (FIPS codes, field descriptions)
  - a small sample of human-readable field descriptions so the LLM can map
    statistical concepts to column codes
  - a sample of FIPS rows so the LLM sees the shape of geography filters

The result is cached with `functools.lru_cache` — one load per process.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from db.snowflake_client import SnowflakeClient, SnowflakeError, get_client

logger = logging.getLogger(__name__)

MAX_SCHEMA_CHARS = 3000

ACS_TABLE_LEGEND = {
    "B01": "age and sex",
    "B02": "race",
    "B03": "Hispanic/Latino origin",
    "B07": "geographic mobility",
    "B08": "commuting",
    "B09": "children",
    "B11": "household type",
    "B12": "marital status",
    "B14": "school enrollment",
    "B15": "education attainment",
    "B16": "language spoken at home",
    "B17": "poverty status",
    "B19": "income",
    "B20": "earnings",
    "B21": "veteran status",
    "B22": "food stamps/SNAP",
    "B23": "employment",
    "B24": "occupation",
    "B25": "housing",
    "B27": "health insurance",
    "B28": "internet access",
    "B29": "citizenship",
    "B99": "allocation/imputation flags",
}

CBG_TABLE_RE = re.compile(r"^(\d{4})_CBG_([A-Z0-9]+)$")


@dataclass(frozen=True)
class SchemaBundle:
    """Structured version of the schema — useful for tests and for the
    string-rendering function below."""

    database: str
    schema: str
    years: list[int]
    table_codes: list[str]
    fips_table: str | None
    field_desc_table: str | None
    geo_table: str | None
    field_description_samples: list[dict[str, Any]]
    fips_samples: list[dict[str, Any]]
    state_codes: list[str] = field(default_factory=list)


def _list_tables(
    client: SnowflakeClient, database: str, schema: str
) -> list[str]:
    sql = (
        "SELECT TABLE_NAME FROM "
        f"{database}.INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = %s "
        "ORDER BY TABLE_NAME"
    )
    rows = client.query(sql, (schema,), max_rows=5000)
    return [r["TABLE_NAME"] for r in rows]


def _pick_latest_metadata_table(
    tables: list[str], suffix: str
) -> str | None:
    """From tables like `2019_METADATA_CBG_FIELD_DESCRIPTIONS` pick the
    newest year that matches the suffix."""
    candidates = []
    for t in tables:
        m = re.match(r"^(\d{4})_" + re.escape(suffix) + r"$", t)
        if m:
            candidates.append((int(m.group(1)), t))
    if not candidates:
        return None
    return max(candidates)[1]


def load_schema_bundle(client: SnowflakeClient | None = None) -> SchemaBundle:
    """Do the live introspection against Snowflake."""
    client = client or get_client()
    database = os.environ.get(
        "SNOWFLAKE_DATABASE",
        "US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET",
    )
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")

    try:
        tables = _list_tables(client, database, schema)
    except SnowflakeError:
        logger.exception("Failed to list tables from Snowflake")
        tables = []

    years: set[int] = set()
    table_codes: set[str] = set()
    for t in tables:
        m = CBG_TABLE_RE.match(t)
        if m:
            years.add(int(m.group(1)))
            table_codes.add(m.group(2))

    fips_table = _pick_latest_metadata_table(tables, "METADATA_CBG_FIPS_CODES")
    field_desc_table = _pick_latest_metadata_table(
        tables, "METADATA_CBG_FIELD_DESCRIPTIONS"
    )
    geo_table = _pick_latest_metadata_table(
        tables, "METADATA_CBG_GEOGRAPHIC_DATA"
    )

    field_description_samples: list[dict[str, Any]] = []
    if field_desc_table:
        try:
            field_description_samples = client.query(
                f'SELECT * FROM {database}.{schema}."{field_desc_table}" '
                f"SAMPLE (25 ROWS)",
                max_rows=25,
            )
        except SnowflakeError:
            logger.warning("Could not sample field descriptions", exc_info=True)

    fips_samples: list[dict[str, Any]] = []
    state_codes: list[str] = []
    if fips_table:
        try:
            fips_samples = client.query(
                f'SELECT * FROM {database}.{schema}."{fips_table}" '
                f"LIMIT 5",
                max_rows=5,
            )
        except SnowflakeError:
            logger.warning("Could not sample FIPS codes", exc_info=True)
        try:
            state_rows = client.query(
                f'SELECT DISTINCT STATE FROM {database}.{schema}.'
                f'"{fips_table}" ORDER BY STATE',
                max_rows=100,
            )
            state_codes = [r["STATE"] for r in state_rows if r.get("STATE")]
        except SnowflakeError:
            logger.warning("Could not enumerate state codes", exc_info=True)

    return SchemaBundle(
        database=database,
        schema=schema,
        years=sorted(years),
        table_codes=sorted(table_codes),
        fips_table=fips_table,
        field_desc_table=field_desc_table,
        geo_table=geo_table,
        field_description_samples=field_description_samples,
        fips_samples=fips_samples,
        state_codes=state_codes,
    )


def render_schema_context(bundle: SchemaBundle) -> str:
    """Turn a SchemaBundle into the compact string the LLM sees."""
    lines: list[str] = []
    lines.append(f"Database: {bundle.database}")
    lines.append(f"Schema: {bundle.schema}")

    if bundle.years:
        lines.append(f"Years available: {', '.join(str(y) for y in bundle.years)}")
    else:
        lines.append("Years available: (none discovered — dataset unavailable)")

    lines.append("")
    lines.append("ACS data tables follow the pattern:")
    lines.append('  {database}.{schema}."<YEAR>_CBG_<CODE>"')
    lines.append("  (table names start with a digit — MUST be double-quoted)")
    lines.append("")
    lines.append("Table code legend:")
    for code in bundle.table_codes:
        desc = ACS_TABLE_LEGEND.get(code, "")
        if desc:
            lines.append(f"  {code}: {desc}")
        else:
            lines.append(f"  {code}")

    lines.append("")
    lines.append("Every CBG data table has a CENSUS_BLOCK_GROUP column "
                 "(12-digit FIPS) plus ACS-code value columns like B01001e1.")

    if bundle.fips_table:
        lines.append("")
        lines.append(
            f"Geography metadata table: \"{bundle.fips_table}\" — one row per "
            "county, keyed on STATE_FIPS + COUNTY_FIPS (NOT CENSUS_BLOCK_GROUP)."
        )
        lines.append(
            "  CBG values in data tables are 12-char strings = "
            "STATE_FIPS(2) + COUNTY_FIPS(3) + TRACT(6) + BLOCKGROUP(1)."
        )
        lines.append(
            "  Join pattern: ON SUBSTR(b.CENSUS_BLOCK_GROUP,1,2)=m.STATE_FIPS "
            "AND SUBSTR(b.CENSUS_BLOCK_GROUP,3,3)=m.COUNTY_FIPS"
        )
        if bundle.fips_samples:
            sample = bundle.fips_samples[0]
            cols = ", ".join(sample.keys())
            lines.append(f"  columns: {cols}")
            lines.append(f"  example row: {sample}")
        if bundle.state_codes:
            lines.append(
                "  STATE values present (all 2-letter postal codes, "
                "including US territories — all IN SCOPE): "
                + ", ".join(bundle.state_codes)
            )

    if bundle.field_desc_table:
        lines.append("")
        lines.append(
            f"Field-description metadata table: \"{bundle.field_desc_table}\" "
            "— maps ACS codes (e.g. B01001e1) to human-readable descriptions."
        )
        if bundle.field_description_samples:
            lines.append("  sample descriptions:")
            for r in bundle.field_description_samples[:10]:
                # These tables commonly have columns TABLE_ID, FIELD_LEVEL_1..N
                # Pull what looks like a code and a description.
                code = (
                    r.get("TABLE_ID")
                    or r.get("FIELD_ID")
                    or r.get("CODE")
                    or next(iter(r.values()), "")
                )
                desc_vals = [
                    v for k, v in r.items()
                    if v and k != "TABLE_ID" and "LEVEL" in k.upper()
                ]
                desc = " > ".join(str(v) for v in desc_vals[:3]) or ""
                line = f"    {code}: {desc}".rstrip(": ")
                lines.append(line)

    if bundle.geo_table:
        lines.append("")
        lines.append(
            f"Geographic lat/lon data: \"{bundle.geo_table}\" "
            "(join on CENSUS_BLOCK_GROUP)."
        )

    text = "\n".join(lines)
    if len(text) > MAX_SCHEMA_CHARS:
        text = text[: MAX_SCHEMA_CHARS - 20] + "\n... (truncated)"
    return text


def build_schema_context() -> str:
    """Alias for get_schema_context() — kept for scripts/docs compatibility."""
    return get_schema_context()


@lru_cache(maxsize=1)
def get_schema_context() -> str:
    """Cached public entry point used by the graph."""
    try:
        bundle = load_schema_bundle()
    except SnowflakeError:
        logger.exception("Schema load failed")
        return (
            "SCHEMA UNAVAILABLE — the Census dataset could not be "
            "introspected. Tell the user there is a temporary data issue."
        )
    return render_schema_context(bundle)


def clear_cache_for_tests() -> None:
    get_schema_context.cache_clear()
    fips_table_name.cache_clear()


@lru_cache(maxsize=1)
def fips_table_name() -> str:
    """Convenience for prompt formatting. Best-effort — falls back to the
    most common name if introspection failed for any reason (Snowflake
    unavailable, driver missing, etc.)."""
    try:
        bundle = load_schema_bundle()
    except Exception:
        logger.warning("fips_table_name fell back to default", exc_info=True)
        return "2019_METADATA_CBG_FIPS_CODES"
    return bundle.fips_table or "2019_METADATA_CBG_FIPS_CODES"
