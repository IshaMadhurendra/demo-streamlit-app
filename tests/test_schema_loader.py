"""Tests for the schema loader: caching, empty-table handling, rendering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from db import schema_loader


@pytest.fixture(autouse=True)
def _clear_cache():
    schema_loader.clear_cache_for_tests()
    yield
    schema_loader.clear_cache_for_tests()


def _make_client(tables: list[str], samples: dict[str, list[dict]] | None = None):
    """Build a mock SnowflakeClient whose `query` returns the right rows
    based on a tiny pattern-match against the SQL."""
    samples = samples or {}
    client = MagicMock()

    def _query(sql, params=None, max_rows=1000):
        if "INFORMATION_SCHEMA.TABLES" in sql:
            return [{"TABLE_NAME": t} for t in tables]
        # Match SAMPLE / LIMIT calls against table name in the SQL.
        for tname, rows in samples.items():
            if f'"{tname}"' in sql:
                return rows
        return []

    client.query.side_effect = _query
    return client


class TestLoadSchemaBundle:
    def test_handles_empty_table_list(self):
        client = _make_client(tables=[])
        bundle = schema_loader.load_schema_bundle(client=client)
        assert bundle.years == []
        assert bundle.table_codes == []
        assert bundle.fips_table is None
        assert bundle.field_desc_table is None

    def test_extracts_years_and_codes(self):
        client = _make_client(tables=[
            "2019_CBG_B01", "2019_CBG_B17", "2020_CBG_B01",
            "2020_METADATA_CBG_FIPS_CODES",
            "2020_METADATA_CBG_FIELD_DESCRIPTIONS",
            "SOMETHING_UNRELATED",
        ])
        bundle = schema_loader.load_schema_bundle(client=client)
        assert bundle.years == [2019, 2020]
        assert "B01" in bundle.table_codes
        assert "B17" in bundle.table_codes
        assert bundle.fips_table == "2020_METADATA_CBG_FIPS_CODES"
        assert bundle.field_desc_table == "2020_METADATA_CBG_FIELD_DESCRIPTIONS"

    def test_picks_latest_metadata_year(self):
        client = _make_client(tables=[
            "2018_METADATA_CBG_FIPS_CODES",
            "2019_METADATA_CBG_FIPS_CODES",
            "2017_METADATA_CBG_FIPS_CODES",
        ])
        bundle = schema_loader.load_schema_bundle(client=client)
        assert bundle.fips_table == "2019_METADATA_CBG_FIPS_CODES"

    def test_snowflake_error_returns_empty_bundle(self):
        from db.snowflake_client import SnowflakeError

        client = MagicMock()
        client.query.side_effect = SnowflakeError("boom")
        bundle = schema_loader.load_schema_bundle(client=client)
        assert bundle.years == []
        assert bundle.fips_table is None


class TestRenderSchemaContext:
    def test_renders_under_3000_chars(self):
        tables = [f"2019_CBG_B{n:02d}" for n in (1, 2, 17, 19, 25)]
        tables += ["2019_METADATA_CBG_FIPS_CODES",
                   "2019_METADATA_CBG_FIELD_DESCRIPTIONS"]
        client = _make_client(tables=tables, samples={
            "2019_METADATA_CBG_FIPS_CODES": [
                {"CENSUS_BLOCK_GROUP": "060014001001",
                 "STATE": "California", "COUNTY": "Alameda County"},
            ],
            "2019_METADATA_CBG_FIELD_DESCRIPTIONS": [
                {"TABLE_ID": "B01001e1", "FIELD_LEVEL_1": "Total population"},
            ] * 10,
        })
        bundle = schema_loader.load_schema_bundle(client=client)
        rendered = schema_loader.render_schema_context(bundle)
        assert len(rendered) < 3000
        assert "California" in rendered
        assert "B01" in rendered
        assert "double-quoted" in rendered

    def test_includes_legend_descriptions(self):
        client = _make_client(tables=["2019_CBG_B19", "2019_CBG_B25"])
        bundle = schema_loader.load_schema_bundle(client=client)
        rendered = schema_loader.render_schema_context(bundle)
        assert "income" in rendered.lower()
        assert "housing" in rendered.lower()


class TestCaching:
    def test_get_schema_context_cached(self, monkeypatch):
        calls = {"n": 0}

        def fake_load():
            calls["n"] += 1
            return schema_loader.SchemaBundle(
                database="d", schema="s", years=[2019], table_codes=["B01"],
                fips_table=None, field_desc_table=None, geo_table=None,
                field_description_samples=[], fips_samples=[],
            )

        monkeypatch.setattr(schema_loader, "load_schema_bundle",
                            lambda client=None: fake_load())
        schema_loader.clear_cache_for_tests()
        a = schema_loader.get_schema_context()
        b = schema_loader.get_schema_context()
        assert a == b
        assert calls["n"] == 1
